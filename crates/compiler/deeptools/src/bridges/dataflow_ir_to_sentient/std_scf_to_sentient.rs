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

//! `SCFToSentient.cpp` — 4 of bridge 2's 384 functions (dependency level(s) [0, 2]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e045_getSentientCmpIPredicate` | 045/384 | 16 | `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:33` |
//! | `e046_ConversionPattern` | 046/384 | 0 | `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:70` |
//! | `e047_runOnOperation` | 047/384 | 30 | `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:251` |
//! | `e225_matchAndRewrite` | 225/384 | 69 | `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:72` |

use crate::arch::Arch;
use crate::islands::dataflow_ir::dialects::arith::CmpIPredicate;
use crate::islands::dataflow_ir::dialects::{self as dfir_dialects, Op as DfirOp, scf};
use crate::islands::dataflow_ir::{self as dfir};
use crate::islands::sentient::dialects::sentient as sen;

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 045/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e045_getSentientCmpIPredicate
///
/// **045/384** `getSentientCmpIPredicate` — `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:33` (16L).
///
/// ```cpp
/// static CmpIPredicate getSentientCmpIPredicate(
///     mlir::arith::CmpIPredicate condop) {
///   if (condop == mlir::arith::CmpIPredicate::eq) {
///     return CmpIPredicate::eq;
///   } else if (condop == mlir::arith::CmpIPredicate::ne) {
///     return CmpIPredicate::ne;
///   } else if (condop == mlir::arith::CmpIPredicate::slt) {
///     return CmpIPredicate::slt;
///   } else if (condop == mlir::arith::CmpIPredicate::sle) {
///     return CmpIPredicate::sle;
///   } else if (condop == mlir::arith::CmpIPredicate::sgt) {
///     return CmpIPredicate::sgt;
///   } else if (condop == mlir::arith::CmpIPredicate::sge) {
///     return CmpIPredicate::sge;
///   } else {
///     llvm_unreachable("invalid predicate");
///   }
/// }
/// ```
///
/// # ⭐⭐ TWO ENUMS THAT SPELL THE SAME SIX WORDS ARE STILL TWO ENUMS
///
/// The argument is `mlir::arith::CmpIPredicate` and the result is `mlir::sentient::CmpIPredicate`
/// (`using namespace mlir::sentient` at `:28` is what makes the unqualified return type the sentient
/// one). They agree on `eq`/`ne`/`slt`/`sle`/`sgt`/`sge` and disagree on everything else: `arith`
/// declares ten enumerators, `SentientTypes.td:474-489` declares six. So this is a narrowing across a
/// rung boundary, not a cast — which is exactly why it is a function and not a `static_cast`.
///
/// # ⛔⛔ `llvm_unreachable` HAS NO INPUT HERE, BY CONSTRUCTION
///
/// The four unsigned predicates `ult`/`ule`/`ugt`/`uge` are the ones that reach the abort. This
/// island's [`CmpIPredicate`] declares the six signed forms and nothing else, *because* of this
/// function and its twin (see that type's own note, which cites entry 048). An unsigned comparison
/// reaching this pipeline is a value the IR cannot hold rather than a run-time stop — and
/// `llvm_unreachable` in a release build is not a stop at all: it is undefined behaviour, so
/// reproducing it as a run-time refusal would be *stricter* than the reference and reproducing it as a
/// fall-through would be a wrong answer.
///
/// # ⚠️ AND THERE ARE TWO COPIES OF THIS FUNCTION IN THE REFERENCE
///
/// Entry 048 is the same if-chain in `StandardToSentient.cpp:36`, file-static in its own pass, whose
/// `else` is `DT_CHECK(0)` followed by a `return CmpIPredicate::eq;` *"to silence to warning"*. Both
/// are scheduled, and each is ported in its own pass's home
/// ([`super::std_standard_to_sentient::get_sentient_cmp_i_predicate`]) rather than shared — one file's
/// copy diverging from the other's is a fact about the reference that a single helper would hide.
#[must_use]
pub const fn get_sentient_cmp_i_predicate(condop: CmpIPredicate) -> sen::CmpPredicate {
    match condop {
        CmpIPredicate::Eq => sen::CmpPredicate::Eq,
        CmpIPredicate::Ne => sen::CmpPredicate::Ne,
        CmpIPredicate::Slt => sen::CmpPredicate::Slt,
        CmpIPredicate::Sle => sen::CmpPredicate::Sle,
        CmpIPredicate::Sgt => sen::CmpPredicate::Sgt,
        CmpIPredicate::Sge => sen::CmpPredicate::Sge,
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 046/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// ONE OF THE THREE REWRITES `SCFToSentientLoweringPass` REGISTERS — the pattern, by its root op.
///
/// ⭐⭐ A PATTERN IS ITS ROOT OP AND ITS BENEFIT, and that is all a `ConversionPattern` constructed
/// this way carries: `ConversionPattern(RootOp::getOperationName(), /*benefit=*/1, ctx)`. All three
/// take benefit 1, so no pattern outranks another — the driver picks by root op alone and there is no
/// ordering question for the port to answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Pattern {
    /// `ForOpLowering` — root `scf.for`, benefit 1 (`SCFToSentient.cpp:69-70`). Its rewrite is
    /// entry 225.
    ForOpLowering,
    /// `IfOpLowering` — root `scf.if`, benefit 1 (`SCFToSentient.cpp:147-148`).
    IfOpLowering,
    /// `YieldOpLowering` — root `scf.yield`, benefit 1 (`SCFToSentient.cpp:237-239`).
    YieldOpLowering,
}

impl Pattern {
    /// THE BENEFIT EVERY ONE OF THEM IS CONSTRUCTED WITH — the `1` in
    /// `ConversionPattern(name, 1, ctx)`.
    pub const BENEFIT: u16 = 1;

    /// THE ROOT OPERATION NAME THE PATTERN IS REGISTERED UNDER.
    ///
    /// ⭐ `mlir::scf::ForOp::getOperationName()` IS THE DIALECT-QUALIFIED MNEMONIC, `"scf.for"` — the
    /// string the driver keys its pattern map on, not the C++ class name.
    #[must_use]
    pub const fn root_op(self) -> &'static str {
        match self {
            Self::ForOpLowering => "scf.for",
            Self::IfOpLowering => "scf.if",
            Self::YieldOpLowering => "scf.yield",
        }
    }
}

/// Replaces: e046_ConversionPattern
///
/// **046/384** `ForOpLowering::ForOpLowering` — `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:70` (0L).
///
/// ```cpp
/// struct ForOpLowering : public mlir::ConversionPattern {
///   explicit ForOpLowering(mlir::MLIRContext *ctx)
///       : mlir::ConversionPattern(mlir::scf::ForOp::getOperationName(), 1, ctx) {}
/// ```
///
/// # ⭐⭐ THE CONSTRUCTOR **IS** THE REGISTRATION, AND ITS BODY IS EMPTY FOR THAT REASON
///
/// A `ConversionPattern` built from `(root_op_name, benefit, context)` matches exactly one operation
/// name at exactly one benefit. So the two facts this scheduled unit contributes are *which op*
/// (`scf.for`) and *at what benefit* (1) — everything else about the rewrite is
/// `matchAndRewrite`, entry 225. There is no state to construct, which is why the reference's body is
/// `{}`.
///
/// # ⛔ THE `MLIRContext *` IS A HANDLE ON THE PATTERN DRIVER, NOT AN INPUT TO THE REWRITE
///
/// It carries the pattern's uniquing and its type/attribute storage. This crate has no pattern driver
/// — [`run_on_operation`] dispatches on the op itself — so the context has nothing to be, and a port
/// that invented one would make the *presence of a compiler session* a value the lowering reads.
///
/// # ⚠️ AND THE ANONYMOUS NAMESPACE AROUND IT IS LOAD-BEARING IN C++ ONLY
///
/// The file's own comment (`SCFToSentient.cpp:52-60`) explains that `YieldOpLowering` also exists at
/// global scope in upstream MLIR, that without `namespace {}` both weak vtables are exported and the
/// loader merges them, and that deeptools objects then get MLIR's incompatible vtable — *"-> SIGSEGV
/// in ~FrozenRewritePatternSet"*. There are no vtables here; the equivalent statement is that
/// [`Pattern`] names dcc's three rewrites and cannot be confused with anybody else's.
#[must_use]
pub const fn for_op_lowering() -> Pattern {
    Pattern::ForOpLowering
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 047/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHAT THE CONVERSION TARGET SAYS ABOUT ONE OP — ⛔ THREE ANSWERS, NOT TWO.
///
/// ⛔⛔ "NOT LEGAL" IS NOT "ILLEGAL", AND A PARTIAL CONVERSION TURNS ON THE DIFFERENCE.
/// `applyPartialConversion` fails only on ops the target marks ILLEGAL and could not rewrite — the
/// pass says so itself: *"The conversion will signal failure if any of our `illegal` operations were
/// not converted successfully"* (`SCFToSentient.cpp:275-277`). An op the target never mentions is left
/// exactly where it is. Collapsing the two would make this pass fail on every `dataflow.*`,
/// `affine.*` and `agen.*` op in the module — which is all of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Legality {
    /// `target.addLegalDialect<...>` — stays, and no pattern is tried on it.
    Legal,
    /// `target.addIllegalDialect<scf::SCFDialect>()` — must be rewritten or the pass fails.
    Illegal,
    /// Named by neither, so a PARTIAL conversion leaves it alone.
    Unmentioned,
}

/// THE CONVERSION TARGET — `SCFToSentient.cpp:261-266`.
///
/// ```cpp
/// target.addLegalDialect<
///     arith::ArithDialect, mlir::vectorchain::VectorChainDialect,
///     mlir::sentient::SentientDialect, mlir::memref::MemRefDialect,
///     mlir::func::FuncDialect>();
///
/// target.addIllegalDialect<scf::SCFDialect>();
/// ```
///
/// ⭐ TWO OF THE FIVE LEGAL DIALECTS HAVE NO ARM HERE. `sentient` is the rung this pass produces —
/// its ops are what the patterns insert, and this island keeps them in a different `Op` type
/// ([`crate::islands::sentient::dialects::Op`]) rather than as a `DfirOp` variant, so a
/// `sentient.*` op is not something this function can be asked about. `memref` and `func` are the
/// module scaffolding (`func.func`, `memref.alloc`) that this crate emits structurally rather than as
/// ops. Their legality is still stated above because dropping a dialect from that list is what would
/// make the pass fail on it.
///
/// ⛔ AND `affine`, `dataflow` AND `agen` ARE ABSENT FROM **BOTH** LISTS. dcc runs this pass inside a
/// pipeline where those rungs are still present; they are [`Legality::Unmentioned`], and a partial
/// conversion is precisely the mode that tolerates them.
#[must_use]
pub const fn legality(op: &DfirOp) -> Legality {
    match op {
        // `addIllegalDialect<scf::SCFDialect>()`.
        DfirOp::Scf(_) => Legality::Illegal,
        // `addLegalDialect<arith::ArithDialect, vectorchain::VectorChainDialect, ...>`.
        DfirOp::Arith(_) | DfirOp::VectorChain(_) => Legality::Legal,
        // Named by neither list. ⭐ `symbol` JOINS THEM: the pass's two lists do not mention it, and
        // a `symbol.create_symbol` does survive this rung — see [`sen::Op::Symbol`].
        // ⭐ AND `vector` JOINS THEM, WHICH THE LIST ABOVE SETTLES BY OMISSION:
        // `addLegalDialect<arith, vectorchain, sentient, memref, func>` names five and
        // `addIllegalDialect<scf>` names one (`SCFToSentient.cpp:261-266`) — `mlir::vector` is in
        // neither, so a `vector.store` sitting beside the `scf.for` this pass rewrites is tolerated
        // by the partial conversion exactly as an `agen` access is.
        //
        // ⭐ AND SO DOES `uniform`, BY THE SAME OMISSION — a `uniform.uniformize_regions` still
        // standing at this rung is left exactly where it is.
        DfirOp::Affine(_)
        | DfirOp::Dataflow(_)
        | DfirOp::Agen(_)
        | DfirOp::Vector(_)
        | DfirOp::Symbol(_)
        | DfirOp::Uniform(_) => Legality::Unmentioned,
    }
}

/// WHICH PATTERN OWNS ONE ILLEGAL OP — `patterns.insert<...>` (`SCFToSentient.cpp:271-273`).
///
/// ⛔⛔ TOTAL OVER THE `scf` DIALECT, WHICH IS WHAT MAKES THE PASS'S FAILURE UNREACHABLE. The three
/// patterns cover `scf.for`, `scf.if` and `scf.yield`; this island's `scf` also declares
/// [`scf::Op::Parallel`], which no pattern matches — so an `scf.parallel` in the module is an illegal
/// op the driver cannot rewrite and `signalPassFailure()` is the reference's answer. That case is
/// reported by the same [`todo!`] as the unported rewrites rather than by a `Result`: the pass either
/// lowers the module or the build stops naming what is missing.
#[must_use]
pub const fn pattern_for(op: &scf::Op) -> Option<Pattern> {
    match op {
        scf::Op::For { .. } => Some(Pattern::ForOpLowering),
        scf::Op::If { .. } => Some(Pattern::IfOpLowering),
        scf::Op::Yield { .. } => Some(Pattern::YieldOpLowering),
        // ⛔ NO PATTERN. `scf.parallel` is illegal and unmatched; see this function's own note.
        scf::Op::Parallel { .. } => None,
    }
}

/// PREORDER, DESCENDING INTO EVERY REGION — what `applyPartialConversion` walks.
///
/// ⛔ A CONVERSION IS NOT A TOP-LEVEL SCAN. The driver legalizes the whole operation it is handed,
/// nested regions included, which is why an `scf.yield` inside an `scf.for` inside a
/// `dataflow.program_unit` is reached. This is the one mechanism the campaign's ports may simplify:
/// MLIR nests through `Region`/`Block`, this island nests through `Vec`s.
///
/// ⛔⛔ THE DESCENT IS [`dfir_dialects::regions`] AND NOT A SECOND MATCH OF ITS OWN. This function
/// once spelled the region-bearing ops out again, and drifted: it lacked the
/// [`dfir_dialects::affine::Op::If`] arm, so an `scf` op in either branch of an `affine.if` was
/// never visited and
/// this pass reported success on a module still holding an op it had declared illegal. `regions`
/// carries the island's own audit note demanding every reader descend through it; the whole point of
/// there being one such function is that a region-bearing op added to the island cannot become
/// invisible to a walk.
fn walk_preorder(ops: &[DfirOp], visit: &mut impl FnMut(&DfirOp)) {
    for op in ops {
        visit(op);
        for region in dfir_dialects::regions(op) {
            walk_preorder(region, visit);
        }
    }
}

/// Replaces: e047_runOnOperation
///
/// **047/384** `SCFToSentientLoweringPass::runOnOperation` —
/// `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:251` (30L).
///
/// ```cpp
/// void SCFToSentientLoweringPass::runOnOperation() {
///   ModuleOp module_op = getOperation();
///   MLIRContext *context = &getContext();
///
///   // The first thing to define is the conversion target. This will define the
///   // final target for this lowering.
///   ConversionTarget target(getContext());
///
///   // We define the specific operations, or dialects, that are legal targets for
///   // this lowering.
///   target.addLegalDialect<
///       arith::ArithDialect, mlir::vectorchain::VectorChainDialect,
///       mlir::sentient::SentientDialect, mlir::memref::MemRefDialect,
///       mlir::func::FuncDialect>();
///
///   target.addIllegalDialect<scf::SCFDialect>();
///
///   // Now that the conversion target has been defined, we just need to provide
///   // the set of patterns that will lower the Toy operations.
///   mlir::RewritePatternSet patterns(context);
///   patterns.insert<ForOpLowering>(context);
///   patterns.insert<IfOpLowering>(context);
///   patterns.insert<YieldOpLowering>(context);
///
///   // With the target and rewrite patterns defined, we can now attempt the
///   // conversion. The conversion will signal failure if any of our `illegal`
///   // operations were not converted successfully.
///   if (failed(applyPartialConversion(module_op, target, std::move(patterns)))) {
///     signalPassFailure();
///   }
/// }
/// ```
///
/// # ⭐⭐ THE PASS IS THREE STATEMENTS: A TARGET, A PATTERN SET, AND ONE DRIVE OF THEM
///
/// [`legality`] is the target; [`PATTERNS`] is the pattern set; the walk below is
/// `applyPartialConversion` over the module. Nothing here decides *how* an `scf.for` becomes a
/// `sentient.for` — that is each pattern's `matchAndRewrite`.
///
/// # ⛔⛔ EVERY ONE OF THE THREE REWRITES IS OUTSIDE THIS BATCH, AND TWO ARE OUTSIDE THE CAMPAIGN
///
/// `ForOpLowering::matchAndRewrite` is entry **225** (`SCFToSentient.cpp:72`, level 2), scheduled and
/// not yet ported. `IfOpLowering::matchAndRewrite` (`:158`) and `YieldOpLowering::matchAndRewrite`
/// (`:241`) appear in neither the 384 nor the campaign's documented exclusions — the extract's
/// per-file census kept each pattern's constructor and only `ForOpLowering`'s rewrite. So this pass
/// cannot lower a module today, and the [`todo!`] names the first `scf` op it meets together with the
/// pattern that owes the rewrite, rather than pretending the pattern set is reachable.
///
/// ⛔ AND THAT IS WHY IT IS WIRED IN ANYWAY. The day an `scf` op reaches this rung the program needs a
/// rewrite that does not exist, and the answer must be a BUILD FAILURE naming it — not a pass that
/// quietly leaves an `scf.for` in a module the Sentient rung cannot schedule.
///
/// # ⛔ WHAT THE PORT DROPS
///
/// - `MLIRContext *context` — the pattern driver's storage; see [`for_op_lowering`].
/// - `std::move(patterns)` — the `FrozenRewritePatternSet` handoff, whose destructor is the SIGSEGV
///   the file's anonymous namespace exists to avoid (`:52-60`).
/// - `signalPassFailure()` — a pass-manager signal, and this crate has no pass manager: which pass
///   runs is a call in [`super::program`]. There is nothing to signal to and nothing to recover.
///
/// # ⛔ AND IT TAKES A SHARED REFERENCE
///
/// Every rewrite behind it is unported, so there is nothing to write back: the whole observable effect
/// is the choice between leaving a module alone and stopping the build. `&mut` would claim a
/// capability nothing behind it has; the parameter becomes `&mut` in the changeset that lands
/// entry 225.
pub fn run_on_operation<A: Arch>(program: &dfir::Program<A>) {
    // ⭐ THE WHOLE MODULE, PREAMBLE INCLUDED. `getOperation()` is the `ModuleOp`, so the units'
    // declarations are as much in scope as their bodies.
    let mut illegal: Option<(Legality, Option<Pattern>)> = None;
    let mut note = |op: &DfirOp| {
        if illegal.is_some() {
            return;
        }
        if let (Legality::Illegal, DfirOp::Scf(scf_op)) = (legality(op), op) {
            illegal = Some((Legality::Illegal, pattern_for(scf_op)));
        }
    };
    walk_preorder(&program.preamble, &mut note);
    for unit in program.units.iter() {
        walk_preorder(&unit.body, &mut note);
    }

    if let Some((_, pattern)) = illegal {
        // ⛔ THE FIRST ILLEGAL OP IS THE ONE THE BUILD MUST NAME, and which pattern owes its rewrite
        // is the actionable half. `None` is an `scf` op no pattern matches at all — the case the
        // reference answers with `signalPassFailure()`.
        todo!(
            "SCFToSentient: an illegal scf op reached this rung; its rewrite is unported \
             (pattern {:?}, root {:?}, of the {} registered)",
            pattern,
            pattern.map(Pattern::root_op),
            PATTERNS.len()
        );
    }
}

/// THE THREE PATTERNS, IN INSERTION ORDER — `SCFToSentient.cpp:271-273`.
///
/// ⭐ ORDER IS RECORDED AND IS NOT A PRIORITY. All three carry [`Pattern::BENEFIT`], and a
/// `RewritePatternSet` is keyed by root op, so insertion order is not what picks a pattern; it is kept
/// because the list itself — *which three rewrites this pass owns* — is the pass's content.
pub const PATTERNS: &[Pattern] = &[
    Pattern::ForOpLowering,
    Pattern::IfOpLowering,
    Pattern::YieldOpLowering,
];

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Target;
    use crate::generated::{OpFunc, SyncSignal};
    use crate::islands::dataflow_ir::dialects::{Val, affine, arith, dataflow};
    use crate::islands::dataflow_ir::ty::IntegerSet;
    use crate::islands::dataflow_ir::{
        Grid, GroupId, OpIndex, Program, ProgramName, ProgramUnit, ProgramUnits, Units,
    };
    use crate::units::DfirUnit;

    /// One unit holding `body` — this pass reads no machine fact, so the kind is arbitrary.
    fn unit_holding(body: Vec<DfirOp>) -> ProgramUnit<Target> {
        ProgramUnit {
            on: Units::one(DfirUnit::Lxlu, Val(0)),
            precision: None,
            body,
            arch: core::marker::PhantomData,
        }
    }

    /// A program of one unit, with `preamble` ahead of it.
    fn program_of(preamble: Vec<DfirOp>, body: Vec<DfirOp>) -> Program<Target> {
        Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            grid: Grid::single(),
            preamble,
            units: ProgramUnits::of(unit_holding(body), Vec::new()),
            arch: core::marker::PhantomData,
        }
    }

    /// 🎯 045/384 — THE SIX SIGNED PREDICATES CROSS THE RUNG UNCHANGED.
    ///
    /// ⛔ AND THEY KEEP THEIR SPELLING, which is what makes the two rungs' text comparable at all:
    /// an input `arith.cmpi sge, ..` becomes a `sentient.if` whose predicate prints `sge`
    /// (`dcc/test/Conversion/StandardToSentient/cmpi_select_different_BB.mlir`). A map that permuted
    /// two predicates would still be total, still exhaustive, and would invert a branch.
    #[test]
    fn the_six_signed_predicates_cross_unchanged() {
        for (arith_pred, sen_pred) in [
            (CmpIPredicate::Eq, sen::CmpPredicate::Eq),
            (CmpIPredicate::Ne, sen::CmpPredicate::Ne),
            (CmpIPredicate::Slt, sen::CmpPredicate::Slt),
            (CmpIPredicate::Sle, sen::CmpPredicate::Sle),
            (CmpIPredicate::Sgt, sen::CmpPredicate::Sgt),
            (CmpIPredicate::Sge, sen::CmpPredicate::Sge),
        ] {
            assert_eq!(get_sentient_cmp_i_predicate(arith_pred), sen_pred);
            assert_eq!(
                get_sentient_cmp_i_predicate(arith_pred).spelling(),
                arith_pred.spelling(),
                "the two rungs spell {arith_pred:?} the same way"
            );
        }
    }

    /// 🎯 045/384 + 048/384 — AND THE REFERENCE'S TWO COPIES OF THE FUNCTION AGREE.
    ///
    /// ⚠️ TWO INDEPENDENT IF-CHAINS IN TWO PASSES (`SCFToSentient.cpp:33` and
    /// `StandardToSentient.cpp:36`), whose `else` arms already differ — `llvm_unreachable` against
    /// `DT_CHECK(0)` plus a fall-through `return CmpIPredicate::eq;`. On the six predicates that
    /// exist they must not differ, or the same `arith.cmpi` lowers two ways depending on which pass
    /// reached it.
    #[test]
    fn both_copies_of_the_predicate_map_agree() {
        for pred in [
            CmpIPredicate::Eq,
            CmpIPredicate::Ne,
            CmpIPredicate::Slt,
            CmpIPredicate::Sle,
            CmpIPredicate::Sgt,
            CmpIPredicate::Sge,
        ] {
            assert_eq!(
                get_sentient_cmp_i_predicate(pred),
                super::super::std_standard_to_sentient::get_sentient_cmp_i_predicate(pred)
            );
        }
    }

    /// 🎯 046/384 — THE PATTERN IS ITS ROOT OP AND ITS BENEFIT.
    ///
    /// ⛔ `getOperationName()` IS THE DIALECT-QUALIFIED MNEMONIC, `"scf.for"` — not `"ForOp"` and not
    /// `"for"`. The driver keys its pattern map on that string, so a pattern registered under
    /// anything else matches nothing and its root op stays illegal.
    #[test]
    fn the_for_pattern_is_registered_on_scf_for_at_benefit_one() {
        assert_eq!(for_op_lowering(), Pattern::ForOpLowering);
        assert_eq!(for_op_lowering().root_op(), "scf.for");
        assert_eq!(Pattern::BENEFIT, 1);
    }

    /// 🎯 046/384 + 047/384 — THREE PATTERNS, THREE DISTINCT ROOT OPS, ALL IN THE `scf` DIALECT.
    ///
    /// ⭐ EQUAL BENEFIT IS WHY THE ROOT OPS MUST BE DISTINCT: two patterns on one op at one benefit
    /// leaves the driver's choice unspecified. And all three roots being `scf.*` is the pass's own
    /// coherence — the one dialect it marks illegal is the one its patterns consume.
    #[test]
    fn the_three_patterns_cover_three_distinct_root_ops() {
        let roots: Vec<&str> = PATTERNS.iter().map(|p| p.root_op()).collect();
        assert_eq!(roots, vec!["scf.for", "scf.if", "scf.yield"]);
        assert!(roots.iter().all(|root| root.starts_with("scf.")));
    }

    /// 🎯 047/384 — THE TARGET MARKS `scf` ILLEGAL, TWO DIALECTS LEGAL, AND THE REST NEITHER.
    ///
    /// ⛔⛔ THE THIRD ANSWER IS THE LOAD-BEARING ONE. `dataflow`, `affine` and `agen` are named by
    /// neither list, and a PARTIAL conversion leaves them alone; were "not legal" the same as
    /// "illegal" this pass would fail on essentially every op in every program this crate emits.
    #[test]
    fn the_conversion_target_is_three_valued() {
        assert_eq!(
            legality(&DfirOp::Scf(scf::Op::Yield {
                operands: Vec::new()
            })),
            Legality::Illegal
        );
        assert_eq!(
            legality(&DfirOp::Arith(arith::Op::Compare {
                result: Val(1),
                predicate: CmpIPredicate::Eq,
                lhs: Val(2),
                rhs: Val(3),
            })),
            Legality::Legal
        );
        assert_eq!(
            legality(&DfirOp::Affine(affine::Op::Yield {
                operands: Vec::new()
            })),
            Legality::Unmentioned
        );
        assert_eq!(
            legality(&DfirOp::Dataflow(dataflow::Op::SyncSend {
                to: Val(0),
                signal: SyncSignal::InputToLxsuToLxluToSync,
            })),
            Legality::Unmentioned
        );
    }

    /// 🎯 047/384 — EVERY `scf` OP A PATTERN COVERS HAS ONE, AND `scf.parallel` HAS NONE.
    ///
    /// ⛔ AN UNMATCHED ILLEGAL OP IS THE REFERENCE'S OWN FAILURE PATH. `scf.parallel` is in the
    /// dialect the target marks illegal and in no pattern's root position, so
    /// `applyPartialConversion` fails on it — which is exactly what this port must not turn into a
    /// silent pass-through.
    #[test]
    fn only_the_three_covered_scf_ops_have_a_pattern() {
        assert_eq!(
            pattern_for(&scf::Op::If {
                cond: Val(0),
                results: Vec::new(),
                body: Vec::new(),
                else_body: Vec::new(),
                dbg_name: None,
            }),
            Some(Pattern::IfOpLowering)
        );
        assert_eq!(
            pattern_for(&scf::Op::Yield {
                operands: Vec::new()
            }),
            Some(Pattern::YieldOpLowering)
        );
        assert_eq!(
            pattern_for(&scf::Op::Parallel {
                ivs: Vec::new(),
                body: Vec::new(),
            }),
            None
        );
    }

    /// 🎯 047/384 — A MODULE WITH NO `scf` OP IS LEFT ALONE, PREAMBLE INCLUDED.
    ///
    /// ⛔⛔ THIS IS THE ONLY THING THE PASS CAN DO TODAY AND IT MUST DO IT WITHOUT COMPLAINT. Every
    /// program this crate emits is `dataflow`/`affine`/`agen`/`arith`/`vectorchain` — all legal or
    /// unmentioned — so a partial conversion over it is a no-op, and a pass that failed on an
    /// unmentioned op would stop every build. REACHING THE END OF THE CALL is the assertion.
    ///
    /// ⛔ IT DOES NOT COMPARE THE PROGRAM WITH A CLONE OF ITSELF. The parameter is shared, so
    /// "unchanged" holds by the type and an equality against a copy would be a tautology dressed as
    /// a check.
    #[test]
    fn a_module_with_no_scf_op_is_left_alone() {
        run_on_operation(&program_of(
            vec![DfirOp::Dataflow(dataflow::Op::SyncSend {
                to: Val(0),
                signal: SyncSignal::InputToLxsuToLxluToSync,
            })],
            vec![
                DfirOp::Arith(arith::Op::Constant {
                    result: Val(1),
                    value: 0,
                }),
                DfirOp::Affine(affine::Op::Yield { operands: vec![] }),
            ],
        ));
    }

    /// 🎯 047/384 — AN `scf` OP REACHING THIS RUNG NAMES THE PATTERN THAT OWES ITS REWRITE.
    ///
    /// ⛔⛔ THIS IS THE WHOLE REASON THE PASS IS WIRED IN WHILE IT CANNOT LOWER ANYTHING. The
    /// alternative to failing here is leaving an `scf.if` in a module the Sentient rung is then asked
    /// to schedule — the reference's `signalPassFailure()` case, silently. The message has to carry
    /// `IfOpLowering`, because "the SCF conversion did not run" is not a diagnosis anybody can act on.
    #[test]
    #[should_panic(expected = "IfOpLowering")]
    fn an_illegal_scf_op_names_its_unported_pattern() {
        run_on_operation(&program_of(
            Vec::new(),
            vec![DfirOp::Scf(scf::Op::If {
                cond: Val(0),
                results: Vec::new(),
                body: Vec::new(),
                else_body: Vec::new(),
                dbg_name: None,
            })],
        ));
    }

    /// 🎯 047/384 — AND THE WALK REACHES AN `scf` OP NESTED IN A REGION IT DOES NOT OWN.
    ///
    /// ⛔⛔ A CONVERSION LEGALIZES THE WHOLE OPERATION IT IS HANDED, REGIONS INCLUDED. Every `scf.if`
    /// this pipeline could produce sits inside an `affine.for` inside a `dataflow.program_unit`; a
    /// top-level scan would find none of them and this pass would report success on a module still
    /// holding the ops it declared illegal.
    #[test]
    #[should_panic(expected = "YieldOpLowering")]
    fn the_walk_descends_into_nested_regions() {
        run_on_operation(&program_of(
            Vec::new(),
            vec![DfirOp::Affine(affine::Op::For {
                iv: Val(0),
                lo: affine::Bound::Const(0),
                hi: affine::Bound::Const(4),
                carried: Vec::new(),
                dbg_name: None,
                body: vec![DfirOp::Scf(scf::Op::Yield {
                    operands: Vec::new(),
                })],
            })],
        ));
    }

    /// A program whose one unit holds an `affine.if` with `body` and `else_body`.
    ///
    /// ⭐ THE SET AND ITS OPERAND ARE ARBITRARY. This pass reads neither — the point is only that the
    /// op OWNS TWO REGIONS, and `affine_set<(d0) : (d0 == 0)>` is the smallest well-formed one.
    fn program_with_affine_if(body: Vec<DfirOp>, else_body: Vec<DfirOp>) -> Program<Target> {
        program_of(
            Vec::new(),
            vec![DfirOp::Affine(affine::Op::If {
                set: IntegerSet::from_sizes(&[1]),
                args: vec![Val(0)],
                symbol_args: Vec::new(),
                results: Vec::new(),
                body,
                else_body,
                dbg_name: None,
            })],
        )
    }

    /// 🎯 047/384 — AND IT REACHES THE `then` BRANCH OF AN `affine.if`.
    ///
    /// ⛔⛔ THIS PINS THE DEFECT THIS REVIEW FOUND. [`walk_preorder`] once matched the region-bearing
    /// ops out again for itself and had drifted: it carried arms for `affine.for`, `scf.for`,
    /// `scf.if`, `scf.parallel`, `dataflow.program_unit` and the composite transfer, and NO arm for
    /// [`affine::Op::If`](crate::islands::dataflow_ir::dialects::affine::Op::If) — whose two
    /// regions are the DataflowIR rung's own conditional
    /// (`dcc/test/PT/issue-236.mlir:59-65`). An `scf` op in either branch was never visited, so this
    /// pass reported success on a module still holding an op it had declared illegal.
    #[test]
    #[should_panic(expected = "YieldOpLowering")]
    fn the_walk_descends_into_the_then_branch_of_an_affine_if() {
        run_on_operation(&program_with_affine_if(
            vec![DfirOp::Scf(scf::Op::Yield {
                operands: Vec::new(),
            })],
            Vec::new(),
        ));
    }

    /// 🎯 047/384 — AND THE `else` BRANCH, WHICH IS THE SECOND REGION AND THE EASIER ONE TO LOSE.
    ///
    /// ⛔ A WALK THAT DESCENDS ONLY THE FIRST REGION PASSES THE `then` TEST ABOVE AND STILL MISSES
    /// HALF OF EVERY CONDITIONAL. [`crate::islands::dataflow_ir::dialects::regions`] yields both,
    /// and that is the only reason both are covered here by the same three lines.
    #[test]
    #[should_panic(expected = "IfOpLowering")]
    fn the_walk_descends_into_the_else_branch_of_an_affine_if() {
        run_on_operation(&program_with_affine_if(
            Vec::new(),
            vec![DfirOp::Scf(scf::Op::If {
                cond: Val(0),
                results: Vec::new(),
                body: Vec::new(),
                else_body: Vec::new(),
                dbg_name: None,
            })],
        ));
    }

    /// 🎯 047/384 — AND AN UNMATCHED ILLEGAL OP STILL STOPS THE BUILD.
    ///
    /// ⛔ `scf.parallel` HAS NO PATTERN AT ALL, so [`pattern_for`] answers `None` — the reference's
    /// `applyPartialConversion` failure rather than an unported rewrite. It is reported by the same
    /// mechanism, because "an illegal op survived" is one outcome however it arose.
    #[test]
    #[should_panic(expected = "None")]
    fn an_illegal_op_with_no_pattern_stops_the_build() {
        run_on_operation(&program_of(
            Vec::new(),
            vec![DfirOp::Scf(scf::Op::Parallel {
                ivs: Vec::new(),
                body: Vec::new(),
            })],
        ));
    }
}

// ⛔ RE-CREATED ANCHORS. These units' `crustify:todo:` markers were deleted without a
// `/// Replaces:` ever appearing, which removed them from every later schedule and let the
// driver report the campaign DONE. Outstanding work is now computed from UNITS.tsv.
// crustify:todo: e225_matchAndRewrite
