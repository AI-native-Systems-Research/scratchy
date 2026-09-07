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

//! `VectorChainToSentientPESFP.cpp` — 8 of bridge 2's 384 functions (dependency level(s) [0, 3, 6, 7, 8]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e076_fuseComputeOps` | 076/384 | 26 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1243` |
//! | `e280_cleanup` | 280/384 | 7 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1056` |
//! | `e344_lowerDanglingNonComputeOpsPESFP` | 344/384 | 93 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1274` |
//! | `e364_patternAgnosticFuseNonComputeOpsHelper` | 364/384 | 222 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:96` |
//! | `e365_fillOpInfo` | 365/384 | 83 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1070` |
//! | `e366_fuseNonComputeOps` | 366/384 | 80 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1159` |
//! | `e377_matchAndRewrite` | 377/384 | 12 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:44` |
//! | `e378_runOnOperation` | 378/384 | 30 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1374` |

use crate::arch::Arch;
use crate::islands::dataflow_ir::dialects::vectorchain as vc;
use crate::islands::dataflow_ir::dialects::{Op as DfirOp, regions};
use crate::islands::dataflow_ir::{self as dfir};
use super::vc_operand_reuse::OperandReuse;

/// ONE OF THE SIXTEEN COMPUTE LOWERING PATTERNS `fuseComputeOps` INSTALLS —
/// `compute_ops_patterns.insert<…>` (`VectorChainToSentientPESFP.cpp:1246-1254`).
///
/// ⭐⭐ SIXTEEN PATTERNS, AND EACH IS A `matchAndRewrite` THAT EMITS A `sentient.compute`. This
/// enumeration is the pass's *table of contents*, not its behaviour: the bodies live at
/// `VectorChainToSentientPESFP.cpp:328` (binary), `:569` (multiply-and-accumulate) and fourteen more,
/// and ⛔ NONE OF THEM IS IN THIS CAMPAIGN'S 384 OR IN ITS 106 DOCUMENTED EXCLUSIONS. Naming them
/// here is what lets [`fuse_compute_ops`] say WHICH lowering a program needs instead of failing
/// anonymously.
///
/// ⛔ THE ESTIMATE FAMILY IS SIX PATTERNS, NOT ONE. `ExpEstimateOpLowering`,
/// `RecEstimateOpLowering`, `LnEstimateOpLowering`, `RsqrtEstimateOpLowering`,
/// `SigmoidEstimateOpLowering` and `TanhEstimateOpLowering` are six separate classes because each
/// picks a different `FEST` mode (`SNComputeLowering.cpp:1316-1372`); this island holds the six as
/// one [`vc::Op::Estimate`] carrying a [`vc::EstimateKind`], so the mapping from op to pattern reads
/// that field.
///
/// ⛔ AND `FastExpOpLowering` IS NOT ONE OF THE SIX. See [`vc::Op::FastExp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ComputePattern {
    /// `BinaryOpLowering` — `VectorChainToSentientPESFP.cpp:328`.
    BinaryOpLowering,
    /// `ElementWiseCompareOpLowering`. ⛔ INSTALLED BUT NOT DECLARED ILLEGAL — see [`legality`].
    ElementWiseCompareOpLowering,
    /// `ElementWiseSelectionOpLowering`. ⛔ INSTALLED BUT NOT DECLARED ILLEGAL.
    ElementWiseSelectionOpLowering,
    /// `MultiplyAndAccumulateOpLowering` — `VectorChainToSentientPESFP.cpp:569`.
    MultiplyAndAccumulateOpLowering,
    /// `MultiplyOpLowering`.
    MultiplyOpLowering,
    /// `ShuffleOpLowering`. ⛔ INSTALLED BUT NOT DECLARED ILLEGAL.
    ShuffleOpLowering,
    /// `PackOpLowering`.
    PackOpLowering,
    /// `ScanWithGapOpLowering`.
    ScanWithGapOpLowering,
    /// `FastExpOpLowering`.
    FastExpOpLowering,
    /// `ExpEstimateOpLowering`.
    ExpEstimateOpLowering,
    /// `FloorOpLowering`.
    FloorOpLowering,
    /// `RecEstimateOpLowering`.
    RecEstimateOpLowering,
    /// `LnEstimateOpLowering`.
    LnEstimateOpLowering,
    /// `RsqrtEstimateOpLowering`.
    RsqrtEstimateOpLowering,
    /// `SigmoidEstimateOpLowering`.
    SigmoidEstimateOpLowering,
    /// `TanhEstimateOpLowering`.
    TanhEstimateOpLowering,
}

/// THE SIXTEEN, IN INSERTION ORDER — `VectorChainToSentientPESFP.cpp:1246-1254`.
///
/// ⭐ THE ORDER IS THE SOURCE'S AND CARRIES NO PRIORITY. `RewritePatternSet::insert` gives every
/// pattern the default benefit of 1 and the driver's order is by benefit then by insertion, so no two
/// of these can ever compete: each matches exactly one op class, and one op has one pattern. The list
/// is in source order so that a diff against the C++ is a diff.
pub const COMPUTE_OPS_PATTERNS: &[ComputePattern] = &[
    ComputePattern::BinaryOpLowering,
    ComputePattern::ElementWiseCompareOpLowering,
    ComputePattern::ElementWiseSelectionOpLowering,
    ComputePattern::MultiplyAndAccumulateOpLowering,
    ComputePattern::MultiplyOpLowering,
    ComputePattern::ShuffleOpLowering,
    ComputePattern::PackOpLowering,
    ComputePattern::ScanWithGapOpLowering,
    ComputePattern::FastExpOpLowering,
    ComputePattern::ExpEstimateOpLowering,
    ComputePattern::FloorOpLowering,
    ComputePattern::RecEstimateOpLowering,
    ComputePattern::LnEstimateOpLowering,
    ComputePattern::RsqrtEstimateOpLowering,
    ComputePattern::SigmoidEstimateOpLowering,
    ComputePattern::TanhEstimateOpLowering,
];

/// A DIALECT THE CONVERSION TARGET DECLARES WHOLLY LEGAL — `target.addLegalDialect<…>`
/// (`VectorChainToSentientPESFP.cpp:1256-1260`).
///
/// ⛔⛔ `vectorchain` IS ABSENT, AND THAT IS THE POINT OF THE PASS: every op of the dialect being
/// converted is a candidate. ⭐ AND `agen` IS ABSENT TOO — unlike `fuseNonComputeOps`, which
/// enumerates the agen, trace and dataflow ops it tolerates one by one (`:1195-1234`). The compute
/// half runs after the non-compute half, by which point the agen ops it would have had to name are
/// already gone.
///
/// ⛔ A DIALECT MISSING FROM THIS LIST IS NOT AN ERROR BY ITSELF. In `applyPartialConversion` an op
/// that is neither legal nor explicitly illegal is offered to the patterns, and left ALONE if none
/// matches — only [`Legality::Illegal`] turns "lower it if you can" into "lower it or fail". That is
/// why a program full of `affine.for`s survives a target that never names the `affine` dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LegalDialect {
    /// `arith::ArithDialect`.
    Arith,
    /// `mlir::sentient::SentientDialect` — what this pass produces.
    Sentient,
    /// `mlir::dataflow::DataflowDialect`.
    Dataflow,
    /// `mlir::memref::MemRefDialect`.
    MemRef,
    /// `mlir::uniform::UniformDialect`.
    Uniform,
    /// `mlir::symbol::SymbolDialect`.
    Symbol,
}

/// THE SIX, IN SOURCE ORDER — `VectorChainToSentientPESFP.cpp:1257-1260`.
pub const LEGAL_DIALECTS: &[LegalDialect] = &[
    LegalDialect::Arith,
    LegalDialect::Sentient,
    LegalDialect::Dataflow,
    LegalDialect::MemRef,
    LegalDialect::Uniform,
    LegalDialect::Symbol,
];

/// WHETHER THE CONVERSION TARGET REQUIRES AN OP TO BE REWRITTEN.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Legality {
    /// Not named by `addIllegalOp`: a pattern may still fire on it, and nothing fails if none does.
    Legal,
    /// `target.addIllegalOp<…>` names it — `applyPartialConversion` must rewrite it or the pass
    /// calls `signalPassFailure()`.
    Illegal,
}

/// WHAT THE TARGET SAYS ABOUT ONE OP — `target.addIllegalOp<…>`
/// (`VectorChainToSentientPESFP.cpp:1261-1264`).
///
/// ⛔⛔ THIRTEEN OPS, AND THE THREE MISSING ONES ARE THE AUDITABLE ASYMMETRY OF THIS PASS.
/// `ElementWiseCompareOp`, `ElementWiseSelectionOp` and `ShuffleOp` all have a pattern in
/// [`COMPUTE_OPS_PATTERNS`] and are **not** in the illegal list, so a program in which their patterns
/// decline to match converts cleanly and keeps those ops. Every other pattern's op is illegal, i.e.
/// its lowering is mandatory. Merging the two lists into one "ops this pass handles" would erase that
/// difference, and with it the reason a stray `vectorchain.shuffle` is a survivable output here and a
/// `vectorchain.binary` is not.
///
/// ⛔ THE SIX ESTIMATES ARE SIX ENTRIES OF THE THIRTEEN. `ExpEstimateOp`, `RecEstimateOp`,
/// `LnEstimateOp`, `RsqrtEstimateOp`, `SigmoidEstimateOp` and `TanhEstimateOp` are six op classes
/// there and one [`vc::Op::Estimate`] here, so every [`vc::EstimateKind`] is illegal and the match
/// says so without naming the field — see [`installed_pattern`], which does read it.
///
/// ⛔ AND THE MATCH IS EXHAUSTIVE OVER `vc::Op` DELIBERATELY. A wildcard would answer `Legal` for the
/// next op somebody adds to the island, silently exempting it from a pass whose whole job is to
/// convert that dialect.
#[must_use]
pub fn legality(op: &DfirOp) -> Legality {
    match op {
        // ── `addIllegalOp<BinaryOp, MultiplyAndAccumulateOp, MultiplyOp, PackOp, ScanWithGapOp,
        //     FastExpOp, ExpEstimateOp, FloorOp, RecEstimateOp, LnEstimateOp, RsqrtEstimateOp,
        //     SigmoidEstimateOp, TanhEstimateOp>()` ─────────────────────────────────────────────
        DfirOp::VectorChain(
            vc::Op::Binary { .. }
            | vc::Op::MultiplyAccumulate { .. }
            | vc::Op::Multiply { .. }
            | vc::Op::Pack { .. }
            | vc::Op::ScanWithGap { .. }
            | vc::Op::FastExp { .. }
            | vc::Op::Floor { .. }
            | vc::Op::Estimate { .. },
        ) => Legality::Illegal,

        // ── the `vectorchain` ops the target does NOT declare illegal ────────────────────────────
        DfirOp::VectorChain(
            vc::Op::ElementWiseCompare { .. }
            | vc::Op::ElementWiseSelection { .. }
            | vc::Op::Shuffle { .. }
            | vc::Op::Select { .. }
            | vc::Op::Merge { .. }
            | vc::Op::ConstantBitstream { .. }
            | vc::Op::Rotate { .. }
            | vc::Op::Cast { .. }
            | vc::Op::CreateAffineMask { .. },
        ) => Legality::Legal,

        // ── every other dialect: `arith` and `dataflow` are declared legal outright, and `affine`,
        //    `scf` and `agen` are simply unnamed, which partial conversion leaves alone ───────────
        DfirOp::Arith(_)
        | DfirOp::Affine(_)
        | DfirOp::Scf(_)
        | DfirOp::Dataflow(_)
        | DfirOp::Agen(_) => Legality::Legal,
    }
}

/// THE PATTERN INSTALLED FOR AN OP, IF ONE IS — the op-class-to-pattern map of
/// `compute_ops_patterns.insert<…>`.
///
/// ⭐ ONE PATTERN PER OP CLASS. Each of the sixteen derives from `OpConversionPattern<XOp>`, so the
/// driver's benefit ordering never arbitrates between two of them; this function is therefore total
/// and single-valued, not a first-match search.
///
/// ⛔ THE ESTIMATE KIND PICKS ONE OF SIX HERE. This is the only place the six-into-one folding of
/// [`vc::Op::Estimate`] has to be undone, and getting it wrong would emit the wrong `FEST` mode —
/// the defect class recorded in `a-port-with-no-caller-is-dead-code`'s precision note.
#[must_use]
pub fn installed_pattern(op: &DfirOp) -> Option<ComputePattern> {
    let DfirOp::VectorChain(op) = op else {
        return None;
    };
    match op {
        vc::Op::Binary { .. } => Some(ComputePattern::BinaryOpLowering),
        vc::Op::ElementWiseCompare { .. } => Some(ComputePattern::ElementWiseCompareOpLowering),
        vc::Op::ElementWiseSelection { .. } => Some(ComputePattern::ElementWiseSelectionOpLowering),
        vc::Op::MultiplyAccumulate { .. } => Some(ComputePattern::MultiplyAndAccumulateOpLowering),
        vc::Op::Multiply { .. } => Some(ComputePattern::MultiplyOpLowering),
        vc::Op::Shuffle { .. } => Some(ComputePattern::ShuffleOpLowering),
        vc::Op::Pack { .. } => Some(ComputePattern::PackOpLowering),
        vc::Op::ScanWithGap { .. } => Some(ComputePattern::ScanWithGapOpLowering),
        vc::Op::FastExp { .. } => Some(ComputePattern::FastExpOpLowering),
        vc::Op::Floor { .. } => Some(ComputePattern::FloorOpLowering),
        vc::Op::Estimate { kind, .. } => Some(match kind {
            vc::EstimateKind::Exp => ComputePattern::ExpEstimateOpLowering,
            vc::EstimateKind::Rec => ComputePattern::RecEstimateOpLowering,
            vc::EstimateKind::Ln => ComputePattern::LnEstimateOpLowering,
            vc::EstimateKind::Rsqrt => ComputePattern::RsqrtEstimateOpLowering,
            vc::EstimateKind::Sigmoid => ComputePattern::SigmoidEstimateOpLowering,
            vc::EstimateKind::Tanh => ComputePattern::TanhEstimateOpLowering,
        }),

        // ⭐ THE FOUR `vectorchain` OPS NO COMPUTE PATTERN CLAIMS. `SelectOp` and `RotateOp` are
        // rearrangements folded into an operand (`Rearrangement::of`, `agen_helper.rs`), `CastOp` is
        // folded into a pack or an on-the-fly precision (`getOperandFromCastOp`, entry 343),
        // `ConstantBitstreamOp` and `CreateAffineMaskOp` become operand fields — and `MergeOp` is
        // handled by `PackOpLowering`'s sibling decoder rather than a pattern of its own.
        vc::Op::Select { .. }
        | vc::Op::Merge { .. }
        | vc::Op::ConstantBitstream { .. }
        | vc::Op::Rotate { .. }
        | vc::Op::Cast { .. }
        | vc::Op::CreateAffineMask { .. } => None,
    }
}

/// WHY ONE OP IN A UNIT IS WORK THIS PORT CANNOT DO YET.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Unlowered {
    /// Illegal AND with a pattern installed: the reference rewrites it, and `signalPassFailure()` if
    /// the rewrite declines. ⭐ ALL THIRTEEN illegal op classes land here, because every one of them
    /// has a pattern — `every_illegal_op_has_an_installed_pattern` is that containment.
    Required(ComputePattern),
    /// A pattern is installed but the op is not declared illegal — the reference tries the rewrite
    /// and leaves the op untouched if the pattern declines. Still an emission this port does not
    /// make, so it is still reported.
    BestEffort(ComputePattern),
    /// ⛔ Illegal with NO pattern installed: `applyPartialConversion` reports *"failed to legalize
    /// operation … that was explicitly marked illegal"*. UNREACHABLE as the two tables are written —
    /// the thirteen illegal classes are a subset of the sixteen patterns' — and named so that a
    /// future edit to one table without the other is a reported condition rather than a silent one.
    Unlegalizable,
}

/// PREORDER, DESCENDING INTO EVERY REGION — `applyPartialConversion(unit_op, …)` walks the whole
/// unit, nested ops included.
///
/// ⛔ IT DESCENDS THROUGH [`regions`], WHICH IS EXHAUSTIVE OVER `Op`, so an op that gains a region
/// gains it here too. A compute inside an `affine.for` body is the ordinary case, not the exception.
fn walk_preorder(ops: &[DfirOp], visit: &mut impl FnMut(&DfirOp)) {
    for op in ops {
        visit(op);
        for region in regions(op) {
            walk_preorder(region, visit);
        }
    }
}

/// WHAT THE PATTERNS WOULD REWRITE IN ONE UNIT, IN PREORDER.
///
/// ⭐ SEPARATE FROM [`fuse_compute_ops`] SO THAT IT IS TESTABLE WITHOUT DIVERGING. The pass itself
/// can only fail on a non-empty answer; the answer is the part with content.
#[must_use]
pub fn compute_ops_to_fuse<A: Arch>(unit: &dfir::ProgramUnit<A>) -> Vec<Unlowered> {
    let mut found: Vec<Unlowered> = Vec::new();
    walk_preorder(&unit.body, &mut |op| {
        match (legality(op), installed_pattern(op)) {
            (Legality::Illegal, Some(pattern)) => found.push(Unlowered::Required(pattern)),
            (Legality::Illegal, None) => found.push(Unlowered::Unlegalizable),
            (Legality::Legal, Some(pattern)) => found.push(Unlowered::BestEffort(pattern)),
            // Legal and unclaimed: the conversion leaves it exactly where it is.
            (Legality::Legal, None) => {}
        }
    });
    found
}

/// Replaces: e076_fuseComputeOps
///
/// FUSE ONE UNIT'S `vectorchain` COMPUTES INTO `sentient.compute`s —
/// `VectorChainToSentientPESFPLoweringPass::fuseComputeOps` (`VectorChainToSentientPESFP.cpp:1243`).
///
/// ```text
///   RewritePatternSet compute_ops_patterns(context);
///   compute_ops_patterns.insert<BinaryOpLowering, …, TanhEstimateOpLowering>(
///       context, dccExtContext(), unit_op, reuse_info);
///   ConversionTarget target(*context);
///   target.addLegalDialect<arith, sentient, dataflow, memref, uniform, symbol>();
///   target.addIllegalOp<BinaryOp, …, TanhEstimateOp>();
///   if (failed(applyPartialConversion(unit_op, target, std::move(compute_ops_patterns)))) {
///     signalPassFailure();
///     return;
///   }
/// ```
///
/// # ⭐⭐ THE FUNCTION IS THREE TABLES AND A DRIVER
///
/// Sixteen patterns ([`COMPUTE_OPS_PATTERNS`]), six legal dialects ([`LEGAL_DIALECTS`]) and thirteen
/// illegal ops ([`legality`]). Everything it *does* is `applyPartialConversion` reading those, which
/// is why the three are named as data here and the driver below is short.
///
/// # ⛔⛔ WHAT IT EMITS IS SIXTEEN `matchAndRewrite` BODIES, AND NOT ONE OF THEM IS SCHEDULED
///
/// `BinaryOpLowering::matchAndRewrite` (`VectorChainToSentientPESFP.cpp:328`),
/// `MultiplyAndAccumulateOpLowering::matchAndRewrite` (`:569`) and fourteen siblings are where the
/// `sentient.compute` is built — with its `operand_a`/`operand_b`/`operand_c` ports, its
/// `op<X>DataID`s from `reuse_info`, its `mode=` and its result forwarding. **None of the sixteen is
/// in the campaign's 384, and none is in its 106 documented exclusions.** So this unit cannot emit,
/// and the honest port is the driver that names the first missing lowering — the same shape
/// `e383_runOnOperation` took for its seven unported tree rewrites.
///
/// # ⛔ AND THE THREE BEST-EFFORT PATTERNS ARE REPORTED TOO
///
/// A `vectorchain.shuffle` is not illegal here, so the reference converts a program containing one
/// whether or not `ShuffleOpLowering` fires. But if it fires, it emits — so "the target tolerates it"
/// is not "this pass does nothing to it". [`Unlowered::BestEffort`] keeps the two apart while
/// reporting both.
///
/// # ⛔ WHAT THE PORT DROPS
///
/// - `MLIRContext *context` and `dccExtContext()`: the pattern set's owner and the target's context.
///   This crate has no `MLIRContext` — patterns are functions, and there is no rewriter to register
///   with. `dccExtContext()` is the machine description the patterns read; it arrives here as `A`.
/// - `std::move(compute_ops_patterns)` into the driver, and `signalPassFailure()`: the failure
///   channel is the panic below, and a pass that failed produced no program.
///
/// # ⭐ `reuse_info` IS TAKEN SHARED, WHICH IS A STATEMENT ABOUT WHAT IS PORTED
///
/// The reference hands it to all sixteen pattern constructors, and those patterns MUTATE it —
/// `OperandReuse::setReuseInformation` latches operands and sets reuse flags. With none of the
/// sixteen ported there is nothing behind the reference to write through it, and a `&mut` here would
/// make every caller hand over exclusive access in order to change nothing. It becomes `&mut` in the
/// changeset that lands the first `matchAndRewrite`.
pub fn fuse_compute_ops<A: Arch>(unit: &dfir::ProgramUnit<A>, reuse_info: &OperandReuse) {
    let to_fuse = compute_ops_to_fuse(unit);

    // ⭐ A UNIT WITH NO COMPUTE IS CONVERTED SUCCESSFULLY AND UNCHANGED — a transfer-only unit
    // reaches `applyPartialConversion` with nothing to legalize and returns `success()`.
    if let Some(first) = to_fuse.first() {
        // ⛔ THE FIRST OP IN PREORDER IS THE ONE THE FAILURE NAMES. Reporting the whole list would
        // be a summary of a pass that never ran; the driver stops at the op it cannot rewrite.
        todo!(
            "e076_fuseComputeOps: {:?} is unported, so {} compute op(s) on {:?} cannot be fused \
             ({:?}); reuse info at entry: {:?}",
            first,
            to_fuse.len(),
            unit.on.kind(),
            &to_fuse[1..],
            reuse_info
        );
    }
}



#[cfg(test)]
mod unit_tests {
    use super::{
        COMPUTE_OPS_PATTERNS, ComputePattern, LEGAL_DIALECTS, Legality, Unlowered,
        compute_ops_to_fuse, fuse_compute_ops, installed_pattern, legality,
    };
    use crate::arch::Target;
    use crate::bridges::dataflow_ir_to_sentient::vc_operand_reuse::OperandReuse;
    use crate::islands::dataflow_ir::dialects::vectorchain as vc;
    use crate::islands::dataflow_ir::dialects::{Op as DfirOp, Val, affine, arith};
    use crate::islands::dataflow_ir::ty::{
        AffineExpr, AffineMap, ElemType, IntegerSet, Vector,
    };
    use crate::islands::dataflow_ir::{ProgramUnit, Units};
    use crate::units::DfirUnit;

    /// The vector every op in these fixtures is typed at.
    const V: Vector = Vector {
        len: 128,
        elem: ElemType::Bf16,
    };

    /// One unit holding `body`.
    fn unit_holding(body: Vec<DfirOp>) -> ProgramUnit<Target> {
        ProgramUnit {
            // ⭐ THE PESFP PASS RUNS ON THE SFP AND THE PE; `sfp` will do.
            on: Units::one(DfirUnit::Sfp, Val(0)),
            precision: None,
            body,
            arch: core::marker::PhantomData,
        }
    }

    fn map() -> AffineMap {
        AffineMap::unary(AffineExpr::Dim(0))
    }

    fn mask() -> vc::Predicate {
        vc::LaneMask::prefix_of(
            128,
            Vector {
                len: 128,
                elem: ElemType::Int(1),
            },
        )
        .binds(Val(90))
    }

    /// ⭐ ONE OF EVERY `vectorchain` OP THIS ISLAND CAN SPELL, with the six estimate kinds spelled
    /// out — twenty-two ops standing for the dialect's twenty-two lowerable forms.
    fn one_of_every_vectorchain_op() -> Vec<DfirOp> {
        let mut ops: Vec<DfirOp> = Vec::new();
        for kind in [
            vc::EstimateKind::Exp,
            vc::EstimateKind::Rec,
            vc::EstimateKind::Ln,
            vc::EstimateKind::Rsqrt,
            vc::EstimateKind::Sigmoid,
            vc::EstimateKind::Tanh,
        ] {
            ops.push(DfirOp::VectorChain(vc::Op::Estimate {
                result: Val(0),
                input: Val(1),
                kind,
                version: None,
                input_ty: V,
                ty: V,
            }));
        }
        ops.push(DfirOp::VectorChain(vc::Op::FastExp {
            result: Val(0),
            input: Val(1),
            input_ty: V,
            ty: V,
        }));
        ops.push(DfirOp::VectorChain(vc::Op::Floor {
            result: Val(0),
            input: Val(1),
            input_ty: V,
            ty: V,
        }));
        ops.push(DfirOp::VectorChain(vc::Op::ScanWithGap {
            result: Val(0),
            input: Val(1),
            reduction_op: vc::BinaryOp::Add,
            input_ty: V,
            ty: V,
        }));
        ops.push(DfirOp::VectorChain(vc::Op::Select {
            result: Val(0),
            input: Val(1),
            selection_map: map(),
            input_ty: V,
            ty: V,
        }));
        ops.push(DfirOp::VectorChain(vc::Op::Multiply {
            result: Val(0),
            a: Val(1),
            b: Val(2),
            reduction_map: map(),
            operand_ty: V,
            ty: V,
        }));
        ops.push(DfirOp::VectorChain(vc::Op::MultiplyAccumulate {
            result: Val(0),
            a: Val(1),
            b: Val(2),
            acc: Val(3),
            reduction_map: map(),
            operand_ty: V,
            ty: V,
        }));
        ops.push(DfirOp::VectorChain(vc::Op::ElementWiseCompare {
            result: Val(0),
            op1: Val(1),
            op2: Val(2),
            mask: None,
            compare_op: vc::CompareOp::Eq,
            operand_ty: V,
            ty: V,
        }));
        ops.push(DfirOp::VectorChain(vc::Op::ElementWiseSelection {
            result: Val(0),
            cond: mask(),
            lhs: Val(1),
            rhs: Val(2),
            mask: None,
            ty: V,
        }));
        ops.push(binary());
        ops.push(DfirOp::VectorChain(vc::Op::Pack {
            result: Val(0),
            op1: Val(1),
            op2: Val(2),
            mask: None,
            indices: vec![0, 1],
            repetition: 1,
            sign_extend: false,
            operand_ty: V,
            ty: V,
        }));
        ops.push(DfirOp::VectorChain(vc::Op::Merge {
            result: Val(0),
            op1: Val(1),
            op2: Val(2),
            iteration_space_ca: IntegerSet {
                dims: 1,
                constraints: Vec::new(),
            },
            access_function_ca: map(),
            access_function_a: map(),
            iteration_space_cb: IntegerSet {
                dims: 1,
                constraints: Vec::new(),
            },
            access_function_cb: map(),
            access_function_b: map(),
            operand_ty: V,
            ty: V,
        }));
        ops.push(DfirOp::VectorChain(vc::Op::ConstantBitstream {
            result: Val(0),
            value: vec![0],
            ty: V,
        }));
        ops.push(DfirOp::VectorChain(vc::Op::Shuffle {
            result: Val(0),
            input: Val(1),
            indices: vec![0, 1],
            repetition: 1,
            input_ty: V,
            ty: V,
        }));
        ops.push(DfirOp::VectorChain(vc::Op::Rotate {
            result: Val(0),
            input: Val(1),
            position: Val(2),
            right_shift: false,
            input_ty: V,
            ty: V,
        }));
        ops.push(DfirOp::VectorChain(vc::Op::Cast {
            result: Val(0),
            input: Val(1),
            input_ty: V,
            ty: V,
        }));
        ops.push(DfirOp::VectorChain(vc::Op::CreateAffineMask {
            result: Val(0),
            mask: vc::LaneMask::prefix_of(
                64,
                Vector {
                    len: 128,
                    elem: ElemType::Int(1),
                },
            ),
        }));
        ops
    }

    /// `vectorchain.binary` — an illegal op with a pattern, the ordinary compute.
    fn binary() -> DfirOp {
        DfirOp::VectorChain(vc::Op::Binary {
            result: Val(0),
            op1: Val(1),
            op2: Val(2),
            mask: None,
            binary_op: vc::BinaryOp::Add,
            op_specific_map: map(),
            operand_ty: V,
            ty: V,
        })
    }

    // ── the three tables ──────────────────────────────────────────────────────────────────────

    /// 🎯 SIXTEEN PATTERNS, ALL DISTINCT — `VectorChainToSentientPESFP.cpp:1246-1254`.
    #[test]
    fn the_pattern_set_holds_the_sixteen_lowerings() {
        assert_eq!(COMPUTE_OPS_PATTERNS.len(), 16);
        for (i, a) in COMPUTE_OPS_PATTERNS.iter().enumerate() {
            for b in &COMPUTE_OPS_PATTERNS[i + 1..] {
                assert_ne!(a, b, "{COMPUTE_OPS_PATTERNS:?}");
            }
        }
    }

    /// 🎯 SIX LEGAL DIALECTS — and neither `vectorchain` nor `agen` is among them (`:1257-1260`).
    #[test]
    fn the_target_declares_six_legal_dialects() {
        assert_eq!(LEGAL_DIALECTS.len(), 6);
        for (i, a) in LEGAL_DIALECTS.iter().enumerate() {
            for b in &LEGAL_DIALECTS[i + 1..] {
                assert_ne!(a, b, "{LEGAL_DIALECTS:?}");
            }
        }
    }

    /// 🎯 THIRTEEN OF THE ISLAND'S TWENTY-TWO LOWERABLE `vectorchain` FORMS ARE ILLEGAL —
    /// `addIllegalOp<…>` names thirteen op classes, six of which are estimates (`:1261-1264`).
    #[test]
    fn thirteen_vectorchain_forms_are_declared_illegal() {
        let illegal = one_of_every_vectorchain_op()
            .iter()
            .filter(|op| legality(op) == Legality::Illegal)
            .count();
        assert_eq!(illegal, 13);
    }

    /// 🎯⛔ EVERY ILLEGAL OP HAS A PATTERN — which is what makes [`Unlowered::Unlegalizable`]
    /// unreachable, and what would break first if one table were edited without the other.
    #[test]
    fn every_illegal_op_has_an_installed_pattern() {
        for op in one_of_every_vectorchain_op() {
            if legality(&op) == Legality::Illegal {
                assert!(installed_pattern(&op).is_some(), "{op:?}");
            }
        }
    }

    /// 🎯⛔ THE ASYMMETRY, NAMED. Three patterns are installed for ops the target does NOT declare
    /// illegal, so a program keeping one of those three still converts.
    #[test]
    fn three_patterns_are_installed_for_ops_that_are_not_illegal() {
        let best_effort: Vec<ComputePattern> = one_of_every_vectorchain_op()
            .iter()
            .filter(|op| legality(op) == Legality::Legal)
            .filter_map(installed_pattern)
            .collect();
        assert_eq!(
            best_effort,
            vec![
                ComputePattern::ElementWiseCompareOpLowering,
                ComputePattern::ElementWiseSelectionOpLowering,
                ComputePattern::ShuffleOpLowering,
            ]
        );
    }

    /// 🎯 THE SIX ESTIMATE KINDS PICK THE SIX ESTIMATE PATTERNS, one each — folding them onto one
    /// pattern would emit the wrong `FEST` mode for five of the six.
    #[test]
    fn each_estimate_kind_names_its_own_pattern() {
        let patterns: Vec<ComputePattern> = one_of_every_vectorchain_op()
            .iter()
            .take(6)
            .filter_map(installed_pattern)
            .collect();
        assert_eq!(
            patterns,
            vec![
                ComputePattern::ExpEstimateOpLowering,
                ComputePattern::RecEstimateOpLowering,
                ComputePattern::LnEstimateOpLowering,
                ComputePattern::RsqrtEstimateOpLowering,
                ComputePattern::SigmoidEstimateOpLowering,
                ComputePattern::TanhEstimateOpLowering,
            ]
        );
    }

    /// 🎯 AND `installed_pattern` COVERS THE WHOLE PATTERN SET: every one of the sixteen is reachable
    /// from some op, so no pattern is named here that the dispatch cannot produce.
    #[test]
    fn every_pattern_in_the_set_is_reachable_from_an_op() {
        let reached: Vec<ComputePattern> = one_of_every_vectorchain_op()
            .iter()
            .filter_map(installed_pattern)
            .collect();
        for pattern in COMPUTE_OPS_PATTERNS {
            assert!(reached.contains(pattern), "{pattern:?}");
        }
    }

    // ── the driver ────────────────────────────────────────────────────────────────────────────

    /// 🎯 A UNIT WITH NO COMPUTE CONVERTS AND CHANGES NOTHING. A transfer-only unit reaches
    /// `applyPartialConversion` with nothing to legalize, and this must not fail.
    #[test]
    fn a_unit_with_no_compute_op_is_left_alone() {
        let unit = unit_holding(vec![
            DfirOp::Arith(arith::Op::Constant {
                result: Val(0),
                value: 0,
            }),
            DfirOp::Affine(affine::Op::Yield {
                operands: Vec::new(),
            }),
        ]);
        assert!(compute_ops_to_fuse(&unit).is_empty());
        fuse_compute_ops(&unit, &OperandReuse::default());
    }

    /// 🎯 THE WALK ENTERS LOOP BODIES — a compute inside an `affine.for` is the ordinary case, and a
    /// top-level-only walk would report a unit full of computes as having none.
    #[test]
    fn a_compute_nested_in_a_loop_is_found() {
        let unit = unit_holding(vec![DfirOp::Affine(affine::Op::For {
            iv: Val(9),
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Const(8),
            carried: Vec::new(),
            body: vec![binary()],
        })]);
        assert_eq!(
            compute_ops_to_fuse(&unit),
            vec![Unlowered::Required(ComputePattern::BinaryOpLowering)]
        );
    }

    /// 🎯 A `vectorchain.cast` IS NEITHER ILLEGAL NOR CLAIMED BY A PATTERN, so the conversion leaves
    /// it where it is and this pass has nothing to report about it.
    #[test]
    fn an_unclaimed_vectorchain_op_is_not_work_for_this_pass() {
        let unit = unit_holding(vec![DfirOp::VectorChain(vc::Op::Cast {
            result: Val(0),
            input: Val(1),
            input_ty: V,
            ty: V,
        })]);
        assert!(compute_ops_to_fuse(&unit).is_empty());
    }

    /// 🎯⛔ AND A `vectorchain.shuffle` IS BEST-EFFORT, NOT REQUIRED — the pattern would fire, but a
    /// unit keeping the op still converts.
    #[test]
    fn a_shuffle_is_reported_as_best_effort() {
        let unit = unit_holding(vec![DfirOp::VectorChain(vc::Op::Shuffle {
            result: Val(0),
            input: Val(1),
            indices: vec![0, 1],
            repetition: 1,
            input_ty: V,
            ty: V,
        })]);
        assert_eq!(
            compute_ops_to_fuse(&unit),
            vec![Unlowered::BestEffort(ComputePattern::ShuffleOpLowering)]
        );
    }

    /// 🎯⛔ AND A REAL COMPUTE FAILS THE BUILD, NAMING ITS LOWERING. ⛔ A pass that silently skipped
    /// the fusion would leave a `vectorchain.binary` in the SentientIR, which the rung below
    /// mis-schedules — the `todo!` is the whole point of wiring this in unported.
    #[test]
    #[should_panic(expected = "BinaryOpLowering")]
    fn a_binary_reaches_the_unported_lowering() {
        fuse_compute_ops(&unit_holding(vec![binary()]), &OperandReuse::default());
    }
}
