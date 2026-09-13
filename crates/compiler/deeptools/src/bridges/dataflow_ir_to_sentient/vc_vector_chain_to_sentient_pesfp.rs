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

use super::vc_operand_reuse::{DataId, OperandReuse};
use super::vc_splat::create_splat_operation;
use super::vc_vector_chain_helper::{
    FusionAnalysis, analyze_and_fill_operand_forwarding, analyze_and_fill_result_forwarding,
    analyze_non_compute_ops_for_fusion, compute_precision_of_op, get_mask_value_for_non_pt,
    input_precision_from_operand, redefine_constant_vectors, result_precision_from_operands,
    vector_type_of,
};
use super::vc_vector_operands::{
    ComputeComp, OpId, OperandValue, VectorOperand, VectorOperandType, defining_position, erase_op,
    erase_op_recording, erase_operands_recording, is_arith_constant, op_at, origin_val, remove_at,
    same_block, use_positions,
};
use crate::arch::{Arch, IsaGen};
use crate::islands::dataflow_ir::dialects::vectorchain as vc;
use crate::islands::dataflow_ir::dialects::{
    Op as DfirOp, Val, agen, dataflow, dbg_name, operands as op_operands, regions, vector,
};
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::dataflow_ir::{self as dfir, Values};
use crate::islands::sentient::dialects::{self as sen, sentient};
use crate::units::DfirUnit;
use std::collections::BTreeMap;

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
            // ⛔ A NEGATION IS LEGAL HERE AND THAT IS NOT AN OVERSIGHT: this pass FOLDS one into the
            // FMA it feeds rather than converting it — `dyn_cast<vectorchain::NegOp>` on the
            // multiply's two inputs, `VectorChainToSentientPESFP.cpp:534-536` — so a `vectorchain.neg`
            // whose consumer took it is already gone, and one whose consumer did not survives the
            // conversion. No `NegOpLowering` exists; see [`installed_pattern`].
            | vc::Op::Neg { .. }
            | vc::Op::Merge { .. }
            | vc::Op::ConstantBitstream { .. }
            | vc::Op::Rotate { .. }
            | vc::Op::Cast { .. }
            | vc::Op::CreateAffineMask { .. }
            | vc::Op::CreateAffineMaskSet { .. },
        ) => Legality::Legal,

        // ── every other dialect: `arith` and `dataflow` are declared legal outright, and `affine`,
        //    `scf` and `agen` are simply unnamed, which partial conversion leaves alone ───────────
        DfirOp::Arith(_)
        | DfirOp::Affine(_)
        | DfirOp::Scf(_)
        | DfirOp::Dataflow(_)
        | DfirOp::Agen(_)
        // ⭐ `vector` IS UNNAMED BY BOTH LISTS — neither `addLegalDialect<arith, sentient, dataflow,
        // memref, uniform, symbol>` (`:1257-1261`) nor the thirteen-op `addIllegalOp` mentions it, and
        // an op a partial conversion never declares illegal survives untouched.
        | DfirOp::Vector(_)
        // ⭐ AND `uniform` IS IN THAT `addLegalDialect` BY NAME (`:1261`), which is the stronger
        // statement of the same outcome: a local region's `query_map` result feeds the compute ops
        // this pass rewrites, so the target declares the dialect legal rather than leaving it unnamed.
        | DfirOp::Uniform(_)
        | DfirOp::Symbol(_) => Legality::Legal,
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
        // ⛔ NO `NegOpLowering` IS INSTALLED. `compute_ops_patterns.insert<…>` never names `NegOp`,
        // and grep over `dcc/src` finds no such class; the negation reaches sentient as the FMA
        // fusion at `:534-536`, not as a pattern of its own.
        vc::Op::Neg { .. } => None,
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
        | vc::Op::CreateAffineMask { .. }
        | vc::Op::CreateAffineMaskSet { .. } => None,
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

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 364/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHERE ONE GROUP OF FUSED SENTIENT OPS GOES — the two kinds of `rewriter.setInsertionPoint*` this
/// fusion makes.
///
/// ⛔⛔ THE SECOND KIND IS NEVER RESTORED, AND THAT IS OBSERVABLE. `setInsertionPointAfter(to)`
/// (`VectorChainToSentientPESFP.cpp:227`) fires inside the to-operand loop for an `sfpring`
/// destination, and the mask constant (`:252`) and the fused compute (`:271`, `:294`) are created
/// AFTER it — so they land behind that send rather than in front of `op`. One position per group is
/// what stops a port quietly re-ordering them back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InsertAt {
    /// `rewriter.setInsertionPoint(op)` (`:180`) — immediately before the send or store being fused.
    Before(OpId),
    /// `rewriter.setInsertionPointAfter(to.value().op_)` (`:227`) — immediately after an `sfpring`
    /// destination's own send.
    After(OpId),
}

/// ONE GROUP OF SENTIENT OPS AND THE INSERTION POINT THEY WERE BUILT AT, in creation order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusedEmission {
    /// Where the group goes.
    pub at: InsertAt,
    /// The ops, in the order the reference creates them.
    pub ops: Vec<sen::Op>,
}

/// WHAT THE NON-COMPUTE FUSION DID — the reference's `LogicalResult` together with the emission and
/// the erasure it owes its caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NonComputeFusion {
    /// Every group, in the order the reference creates them. ⛔ TWO GROUPS CAN NAME THE SAME
    /// POSITION — an `sfpring` destination puts the `SetSendDst` and then the compute after the same
    /// send — and a caller must splice them IN THIS ORDER, because the reference's insertion point
    /// advances past every op it creates.
    pub emitted: Vec<FusedEmission>,
    /// `erased_list` as the two `VectorOperand::eraseOperands` calls leave it (`:308-312`), sorted
    /// and deduped — the list is only ever asked `contains`, so its order is not part of it.
    ///
    /// ⛔⛔ RETURNED, NOT APPLIED, AND NOT FOR CONVENIENCE. Every position in [`Self::emitted`] names
    /// an op ON THIS LIST — the send being fused, or the `sfpring` send behind it — because a fused
    /// compute REPLACES them. The reference's rewriter holds `Operation *`s and can insert then
    /// erase; an [`OpId`] is a POSITION, so a caller must splice the emission in first and only then
    /// remove these, descending, exactly as [`cleanup`] does.
    pub to_erase: Vec<OpId>,
    /// `return success()` / `return failure()`, and which one.
    pub outcome: FusionOutcome,
}

/// WHY THE NON-COMPUTE FUSION STOPPED, each arm a different fact about the program that reached it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionOutcome {
    /// `return success()` with the whole body skipped (`:101-103`, `:320`) — there is no producer, or
    /// it is not one of the four this fuses. ⛔ NOT A FAILURE: the op stays, and entry 344 is what
    /// lowers what is left of it.
    NotFused,
    /// `return success()` after the rewrite.
    Fused,
    /// `if (!is_fusion_respected) return failure();` (`:114`) — ⛔ BEFORE `setReuseInformation`, so
    /// the reuse map is untouched on this arm.
    FusionNotRespected,
    /// `emitError("All to operands should be in the same block in order to be fused.")` (`:313-317`)
    /// — ⛔ AFTER `setReuseInformation`, whose writes stand.
    NotSameBlock,
    /// `createSplatOperation` answered `failure()` (`:288-291`) — entry 279's own refusal.
    SplatRefused,
    /// THE REFERENCE'S OWN ABORTS, which are not `emitError`s: every
    /// `symbolizeSentientComputePort(…).value()` and `symbolizeSentientPrecision(…).value()`
    /// (`:274-285`, `:296-305`) on a port or precision with no spelling — an EMPTY `opA`, `opC` or
    /// result precision is the one that is reachable, and `getSentientFoldModeAttrForOperation`'s own
    /// `llvm_unreachable` (`Sentient/Utils.cpp:450-451`) arrives here too.
    Unrepresentable,
}

/// Replaces: e364_patternAgnosticFuseNonComputeOpsHelper
///
/// **364/384** `patternAgnosticFuseNonComputeOpsHelper` — `VectorChainToSentientPESFP.cpp:96`
/// (222L): a receive, load or constant feeding a send or store becomes ONE fused compute.
///
/// ⛔ `use_logical = !(use_fma || use_immcopy)` (`:178`), so its two `if`s ARE the others' `else` and
/// the reference's fourth arm is dead; the FMA arm's three `std::swap`s sit INSIDE the to-operand
/// loop and cancel on a second one; `opC_forwarding` (`:182`) is declared and never pushed to.
#[allow(clippy::too_many_arguments)]
pub fn pattern_agnostic_fuse_non_compute_ops_helper<A: Arch>(
    op: &OpId,
    from: Option<&VectorOperand>,
    unit: &OpId,
    comp: ComputeComp,
    reuse_info: &mut OperandReuse,
    is_visited: &mut BTreeMap<OpId, bool>,
    is_precision_converted_global: bool,
    scope: &[DfirOp],
    values: &mut Values,
) -> NonComputeFusion {
    let refused = |outcome| NonComputeFusion {
        emitted: Vec::new(),
        to_erase: Vec::new(),
        outcome,
    };

    // `if (from.has_value() && isa<dataflow::ReceiveOp, vector::LoadOp, agen::VectorLoadOp,
    //  mlir::arith::ConstantOp>(from.value().op_))`, whose `else` is the `return success()` at `:320`.
    let Some(from) = from else {
        return refused(FusionOutcome::NotFused);
    };
    let producer = op_at(&from.op, scope);
    let fuses = matches!(
        producer,
        Some(
            DfirOp::Dataflow(dataflow::Op::Receive { .. })
                | DfirOp::Vector(vector::Op::Load { .. })
                | DfirOp::Agen(agen::Op::VectorLoad { .. })
        )
    ) || is_arith_constant(&from.op, scope);
    if !fuses {
        return refused(FusionOutcome::NotFused);
    }
    // The op being fused as an OP — `getComputePrecisionOfOp`, `getSentientFoldModeAttrForOperation`,
    // `getDbgNameAttr` and `createSplatOperation` all take it that way. An `Operation *` cannot
    // dangle in the reference; a position naming no op is [`FusionOutcome::Unrepresentable`].
    let Some(fused) = op_at(op, scope) else {
        return refused(FusionOutcome::Unrepresentable);
    };

    // `analyzeNonComputeOpsForFusion(unit_, dcc_ext_ctx_, reuse_info_, op, from, comp_, is_visited_,
    //  is_fusion_respected, to_operands, is_dangling_ops_present_after_fusion);` — entry 341, which
    // also carries the three locals the reference declares above it (`:105-107`).
    let mut analysis = FusionAnalysis::default();
    analyze_non_compute_ops_for_fusion::<A>(unit, op, from, comp, is_visited, scope, &mut analysis);

    // `if (!is_fusion_respected) return failure();`
    if !analysis.is_fusion_respected {
        return refused(FusionOutcome::FusionNotRespected);
    }

    // `from_operands.push_back(from); this->reuse_info_.setReuseInformation(op, from_operands);`
    // ⛔ IT IS THE **COPY** THAT GETS LATCHED, and `:185` below reads that copy back while `:122`
    // reads the original `from` — see the two reads' own notes.
    let mut from_operands = vec![from.clone()];
    reuse_info.set_reuse_information(op, &mut from_operands, scope);

    // `if (VectorOperand::sameBlock(op, to_operands).succeeded())`, whose `else` is the `emitError` at
    // `:313-317`. ⛔ THE LIST OVERLOAD (`VectorOperands.cpp:670-685`) DUPLICATES the single-operand
    // one's body rather than delegating to it — an empty list succeeds, and an ABSENT operand fails.
    if !analysis
        .to_operands
        .iter()
        .all(|to| same_block(op, to.as_ref(), scope))
    {
        return refused(FusionOutcome::NotSameBlock);
    }
    // ⭐ WHICH IS WHY THE UNGUARDED `to.value()`s AT `:160` AND `:211` CANNOT FIRE: `sameBlock`'s
    // `else { return failure(); }` (`VectorOperands.cpp:679-681`) already refused an absent operand, so
    // this `flatten` drops nothing and the lengths `:154` and `:174` test are the same lengths.
    let to_operands: Vec<&VectorOperand> = analysis.to_operands.iter().flatten().collect();

    // `std::string op1 = from.value().getName(), op2 = "one", op3 = "zero";` — ⛔ THE ORIGINAL `from`.
    let mut op1 = from.name();
    let op2 = sentient::Port::One;
    let mut op3 = Some(sentient::Port::Zero);
    // `int from_ID[3] = {this->reuse_info_.getId(from.value().op_).value(), -1, -1};` — ⛔ READ AFTER
    // `setReuseInformation`, which is the call that assigned it. `None` is the `.td`'s -1, and the
    // reference's `.value()` is total: `getId` answers -1 on a miss (`OperandReuse.cpp:65-71`).
    let mut from_id: [Option<i32>; 3] = [
        match origin_val(&from.op, scope).map_or(DataId::Unassigned, |val| reuse_info.id(val)) {
            DataId::Unassigned => None,
            assigned => Some(assigned.attribute()),
        },
        None,
        None,
    ];

    // `bool use_fma = false;` and `bool sen1p5_receive_from_pt = isa<ReceiveOp>(from.value().op_) &&
    //  from.value().getFirstValue() == "pt" && dcc_ext_ctx_.getArch() >= IsaCoreGen::SEN1P5_ISA;`
    // ⛔ `getFirstValue()`, NOT `getName()`: the raw first value, which is `pt` for the PT FIFO and a
    // bare slice index for an `lrf`/`irf`/`istate` operand.
    let mut use_fma = false;
    let sen1p5_receive_from_pt = matches!(
        producer,
        Some(DfirOp::Dataflow(dataflow::Op::Receive { .. }))
    ) && from.values.first()
        == Some(&OperandValue::Port(sentient::Port::Pt))
        && A::GEN >= IsaGen::Sen1p5;

    // `if (is_any_of(this->comp_, PE, SFP) && !to_operands.empty())`
    if matches!(comp, ComputeComp::Pe | ComputeComp::Sfp) && !to_operands.is_empty() {
        // `if (is_precision_converted_global) use_fma = true;`
        if is_precision_converted_global {
            use_fma = true;
        }

        // `if (isa<mlir::arith::ConstantOp>(from.value().op_))` — the constant+send case.
        if is_arith_constant(&from.op, scope) {
            for to in &to_operands {
                // `if (to.value().type_ == Link) { use_fma = true; op1 = "lrf14"; }` — the reference's
                // own comment on that port is *"doesn't influence the result"*.
                if to.kind == VectorOperandType::Link {
                    use_fma = true;
                    op1 = Some(sentient::Port::Lrf(sentient::LrfIndex::L14));
                }
            }
        } else if sen1p5_receive_from_pt && comp == ComputeComp::Pe {
            // *"PT FIFO not valid for LOGICAL in sen1p5 so need to realize as FMA."*
            use_fma = true;
        }
    }

    // `bool use_immcopy = (isa<mlir::arith::ConstantOp>(from.value().op_) &&
    //  is_any_of(this->comp_, PE, SFP) && to_operands.size() == 1 &&
    //  to_operands[0].value().type_ == LRF);` — *"Use IMMCOPY if its constant storing into a register."*
    let use_immcopy = is_arith_constant(&from.op, scope)
        && matches!(comp, ComputeComp::Pe | ComputeComp::Sfp)
        && to_operands.len() == 1
        && to_operands[0].kind == VectorOperandType::Lrf;

    // `bool use_logical = !(use_fma || use_immcopy);` — ⛔ NOT A LOCAL HERE, because it is the
    // NEGATION of the other two: both of its `if`s are the `else` of an `if use_fma … else if
    // use_immcopy` chain, and the reference's fourth arm (`llvm::errs() << "Unknown option to lower
    // to sentient"`, `:246-249`) is dead code there is nothing to write as.

    // `rewriter.setInsertionPoint(op);`
    let mut at = InsertAt::Before(op.clone());
    // `SmallVector<Attribute, 1> opA_forwarding, opC_forwarding, result_forwarding;` — ⛔ THE C ONE IS
    // DECLARED AND NEVER PUSHED TO (`:182`), so the MAC's C forwarding is always empty.
    let mut opa_forwarding: Vec<sentient::Port> = Vec::new();
    let opc_forwarding: Vec<sentient::Port> = Vec::new();
    let mut result_forwarding: Vec<sentient::Port> = Vec::new();

    // `std::string opA_precision = getInputPrecisionFromOperand(from_operands[0]);` — ⛔ THE LATCHED
    // COPY, not `from`. It reads `orig_precision_`, which latching does not touch, so the two agree
    // for a one-operand list — but the reference reads the copy and so does this.
    let mut opa_precision = input_precision_from_operand(&from_operands[0]);
    // `auto compute_precision = getComputePrecisionOfOp(op);`
    let mut compute_precision = compute_precision_of_op(fused);
    // `if (sen1p5_receive_from_pt && this->comp_ == PE) { if (compute_precision == "fp16")
    //  compute_precision = "fp32"; }` — *"fp16 coming from PT is really fp24, which shouldn't be
    // downcast to fp16."*
    if sen1p5_receive_from_pt
        && comp == ComputeComp::Pe
        && compute_precision == sentient::Precision::Fp16
    {
        compute_precision = sentient::Precision::Fp32;
    }

    // `if (opA_precision == "fp32" && compute_precision == "fp16")` — *"Use the operation
    // corresponding to the higher of opA/compute precisions to determine foldMode."*
    let fp32_in_fp16_out = opa_precision == Some(sentient::Precision::Fp32)
        && compute_precision == sentient::Precision::Fp16;
    // ⛔ THE FOLD MODE COMES OFF THE **PRODUCER** ON THAT ARM (`:196-198`) AND OFF `op` OTHERWISE
    // (`:204-206`) — the two ops need not carry the same vector type, so this is not one call.
    let Some(fold_of) = (if fp32_in_fp16_out {
        producer
    } else {
        Some(fused)
    }) else {
        return refused(FusionOutcome::Unrepresentable);
    };
    let fold_mode = match fold_mode_attr_for_operation(fold_of, comp, sen1p5_receive_from_pt) {
        FoldModeAttr::Absent => None,
        FoldModeAttr::Present(mode) => Some(mode),
        FoldModeAttr::Unsupported => return refused(FusionOutcome::Unrepresentable),
    };
    if fp32_in_fp16_out {
        // `std::swap(opA_precision, compute_precision);` — *"Cast op should be interpreted as output
        // on the fly conversion, so opA_precision should be fp16 and compute&result precisions should
        // be fp32."* ⛔ THE GUARD PINS BOTH SIDES, so this pair is the whole exchange.
        opa_precision = Some(sentient::Precision::Fp16);
        compute_precision = sentient::Precision::Fp32;
    }
    // `std::string opB_precision = compute_precision; std::string opC_precision = compute_precision;`
    let opb_precision = compute_precision;
    let mut opc_precision = Some(compute_precision);

    let mut emitted: Vec<FusedEmission> = Vec::new();
    // `for (auto &to : to_operands) {`
    for to in &to_operands {
        // `auto destination_type = to.value().type_; std::string dest = to.value().getName();`
        let destination_type = to.kind;
        let Some(dest) = to.name() else {
            return refused(FusionOutcome::Unrepresentable);
        };
        if use_fma {
            // *"Use the accumulation part if plan to use FMA / Solves INT24 challenges in IMA8 /
            // Allows constants 0, 1, 2, 3."* ⛔ ALL THREE SWAPS ARE INSIDE THIS LOOP, so a second
            // to-operand puts every one of them back.
            core::mem::swap(&mut op1, &mut op3);
            core::mem::swap(&mut opa_precision, &mut opc_precision);
            from_id.swap(0, 2);
            result_forwarding.push(dest);

            // `if (dest == "sfpring")` — *"A special case in DD1a where sfp unit forwarding one of its
            // input operands is forwarded to another sfp via MAC. In case of SFPRing, only FMA result
            // can be sent to data fifo."*
            if dest == sentient::Port::SfpRing {
                // `rewriter.setInsertionPointAfter(to.value().op_);` — ⛔ AND NEVER BACK; see
                // [`InsertAt`].
                at = InsertAt::After(to.op.clone());
                // `auto send_op = llvm::dyn_cast<dataflow::SendOp>(to.value().op_);
                //  sentient::SetSendDestinationOp::create(rewriter, op->getLoc(),
                //  send_op.getToUnit());` — ⛔ THE `dyn_cast` IS UNCHECKED, and what makes it safe is
                // that a destination can only ever be a send or a store
                // (`VectorChainHelper.cpp:619-621`, `:671-674`) — ⛔ NOT that only a send names
                // `sfpring`: the RECEIVE side names it too (`VectorOperands.cpp:70-76`, `link +=
                // "ring"` at `:74`; the send side is `:121-129`).
                let Some(DfirOp::Dataflow(dataflow::Op::Send { to: units, .. })) =
                    op_at(&to.op, scope)
                else {
                    return refused(FusionOutcome::Unrepresentable);
                };
                emitted.push(FusedEmission {
                    at: at.clone(),
                    ops: vec![sen::Op::Sentient(sentient::Op::SetSendDst {
                        units: *units,
                    })],
                });
            }
        } else if use_immcopy {
            // `DT_CHECK(destination_type == LRF);` — ⛔ NOT A TEST THIS PORT CAN FAIL: `use_immcopy`
            // IS that comparison, on the single to-operand there is (`:172-175`). ⭐ AND THIS ARM
            // NEVER SYMBOLIZES `dest`, so the refusal above it is stricter than the reference by
            // nothing: an `LRF` operand always spells `lrf<N>`.
        } else {
            // `else if (use_logical) { if (destination_type == LRF) …` — *"LOGICAL doesn't allow
            // src0/src2 port forwarding in tgtrf"*, so a register destination goes on the RESULT and
            // everything else on opA.
            if destination_type == VectorOperandType::Lrf {
                result_forwarding.push(dest);
            } else {
                opa_forwarding.push(dest);
            }
            // *"Precision of opC (0.0) should match opA's precision in LOGICAL."*
            opc_precision = opa_precision;
        }
    }

    // `sentient::ConstantOp::create(rewriter, op->getLoc(), rewriter.getIndexType(), 0);` — ⛔ BUILT
    // BEFORE THE BRANCH, so the immcopy arm emits it and never reads it; entry 279 mints its own.
    let mask = values.mint();
    let mask_const = sen::Op::Sentient(sentient::Op::ScalarConstant {
        value: 0,
        result: mask,
        reg_locale: sentient::RegType::Imm,
        ty: ScalarTy::Index,
        is_symbol: false,
    });

    // `auto op_dbg_name = dataflow::getDbgNameAttr(op);`
    let op_dbg_name = dbg_name(fused).map(str::to_owned);

    // `auto result_precision = getResultPrecisionFromOperands(to_operands);` — entry 061, which reads
    // the FIRST PRESENT operand's `on_the_fly_conv_precision_` and returns its emptiness too. The
    // `DT_CHECK_MSG(from_operands.size() == 1, …)` beside it (`:258`) is the one-element vector this
    // function pushed at `:117`.
    let mut result_precision = result_precision_from_operands(&analysis.to_operands);
    // `if (result_forwarding.empty() && result_precision == "") result_precision = compute_precision;`
    if result_forwarding.is_empty() && result_precision.is_none() {
        result_precision = Some(compute_precision);
    }

    let mut ops = vec![mask_const];
    if use_fma {
        // ⛔ THE FIVE `.value()`s THIS ARM TAKES, and no others: an empty precision or an unspellable
        // port is where the reference dies.
        let (
            Some(op1),
            Some(op3),
            Some(opa_precision),
            Some(opc_precision),
            Some(result_precision),
        ) = (op1, op3, opa_precision, opc_precision, result_precision)
        else {
            return refused(FusionOutcome::Unrepresentable);
        };
        // `sentient::MacOp::create(rewriter, op->getLoc(), TypeRange(), mask_const_op.getResult(),
        //  ValueRange(pointers), op_dbg_name, op1, op2, op3, opA_forwarding, {}, opC_forwarding,
        //  result_forwarding, fold_mode_attr, opA_precision, opB_precision, opC_precision,
        //  result_precision, compute_precision, from_ID[0], from_ID[1], from_ID[2]);`
        ops.push(sen::Op::Sentient(sentient::Op::VectorMac {
            // *"PE/SFP associates mask directly to operations. … Therefore the mask should be set to
            // 0 (default)."*
            mask: Some(mask),
            // `ArrayRef<Value> pointers = {}` — always empty here; the PT pass is what threads one.
            xrf_write_ptr: None,
            xrf_read_ptr: None,
            // `TypeRange()` — a fused non-compute binds nothing.
            results: Vec::new(),
            op_a: sentient::Operand {
                forwarding: opa_forwarding,
                precision: opa_precision,
                data_id: from_id[0],
                ..sentient::Operand::from(op1)
            },
            // `ArrayAttr::get(context, {})` for B's forwarding.
            op_b: sentient::Operand {
                precision: opb_precision,
                data_id: from_id[1],
                ..sentient::Operand::from(op2)
            },
            op_c: sentient::Operand {
                forwarding: opc_forwarding,
                precision: opc_precision,
                data_id: from_id[2],
                ..sentient::Operand::from(op3)
            },
            result: sentient::ResultPorts {
                forwarding: result_forwarding,
                precision: result_precision,
                unroll_incr: false,
            },
            // Not passed to `MacOp::create`, so the `.td` defaults stand.
            mode: sentient::FmaMode::FusedMulAdd,
            compute_precision,
            fold_mode,
            unroll_factor: sentient::UnrollFactor::X1,
            xrf_read_incr: 0,
            xrf_write_incr: 0,
            data_transfer_only: false,
            is_data_weight: None,
            dbg_name: op_dbg_name,
        }));
    } else if use_immcopy {
        // `if (failed(vectorchain::createSplatOperation(op, from.value(), to_operands[0].value(),
        //  rewriter, this->comp_, dcc_ext_ctx_.dsc_global_->sysDef))) return failure();` — entry 279.
        let Some(splat) =
            create_splat_operation::<A>(fused, from, to_operands[0], comp, scope, values)
        else {
            return refused(FusionOutcome::SplatRefused);
        };
        ops.extend(splat);
    } else {
        let (
            Some(op1),
            Some(op3),
            Some(opa_precision),
            Some(opc_precision),
            Some(result_precision),
        ) = (op1, op3, opa_precision, opc_precision, result_precision)
        else {
            return refused(FusionOutcome::Unrepresentable);
        };
        // `sentient::BinaryOp::create(rewriter, op->getLoc(), mask_const_op, op_dbg_name, op1, op3,
        //  SentientBinaryOperator::or0, opA_forwarding, {}, result_forwarding, nullptr,
        //  fold_mode_attr, opA_precision, opC_precision, result_precision, "none", from_ID[0],
        //  from_ID[2]);`
        ops.push(sen::Op::Sentient(sentient::Op::VectorBinary {
            mask,
            op_a: sentient::Operand {
                forwarding: opa_forwarding,
                precision: opa_precision,
                data_id: from_id[0],
                ..sentient::Operand::from(op1)
            },
            // ⛔⛔ **B TAKES `op3`, `opC_precision` AND `from_ID[2]`** — a binary has two operands, so
            // the C slot's port and precision are what land on B and the B ones are dropped.
            op_b: sentient::Operand {
                precision: opc_precision,
                data_id: from_id[2],
                ..sentient::Operand::from(op3)
            },
            // `SentientBinaryOperator::or0` with `nullptr` for the logical result forwarding — the
            // comment block at `:129-139` is why: *"tgtrf = src0 --> not allowed! we need OR with
            // zero."*
            binary_op: sentient::Binary::Plain(sentient::BinaryOp::Or),
            result: sentient::ResultPorts {
                forwarding: result_forwarding,
                precision: result_precision,
                unroll_incr: false,
            },
            // *"Bitwise operation so set compute precision to none."*
            compute_precision: sentient::Precision::None,
            fold_mode,
            unroll_factor: sentient::UnrollFactor::X1,
            dbg_name: op_dbg_name,
        }));
    }
    emitted.push(FusedEmission { at, ops });

    // `std::vector<mlir::Operation *> erased_list;
    //  VectorOperand::eraseOperands(to_operands, rewriter, erased_list);`
    let mut to_erase: Vec<OpId> = Vec::new();
    erase_operands_recording(&analysis.to_operands, scope, &mut to_erase);
    // `if (!is_dangling_ops_present_after_fusion) VectorOperand::eraseOperands(from_operands, …);`
    // ⛔ ONE `erased_list` SPANS BOTH, which is what keeps an op reached from either side claimed
    // once — the same reason [`cleanup`] shares its list across three phases.
    if !analysis.is_dangling_ops_present_after_fusion {
        let from_side: Vec<Option<VectorOperand>> = from_operands.into_iter().map(Some).collect();
        erase_operands_recording(&from_side, scope, &mut to_erase);
    }
    to_erase.sort_unstable();
    to_erase.dedup();

    // `return success();`
    NonComputeFusion {
        emitted,
        to_erase,
        outcome: FusionOutcome::Fused,
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 377/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e377_matchAndRewrite
///
/// `SendOpLowering::matchAndRewrite` (`VectorChainToSentientPESFP.cpp:44`) — the non-compute fusion
/// pattern for a `dataflow.send`: read the operand that PRODUCES the sent vector, then hand the send
/// and that operand to the pattern-agnostic fusion helper.
///
/// ⛔ THE OPERAND IS THE SEND'S PRODUCER, NOT THE SEND (`:48-50`): `getSendData().getDefiningOp()`.
/// Passing the send itself would classify the wrong op — `patternAgnosticFuseNonComputeOpsHelper`
/// fuses only when `from` is a `dataflow.receive`, a `vector.load`, an `agen.vector_load` or an
/// `arith.constant` (`:101-103`), and a send is none of them.
/// ⛔ `is_precision_converted_global` IS BY VALUE (`:99`), so the helper's copy is what the helper
/// reads; nothing here observes a write back, and the local it was copied from (`:47`, set by entry
/// 304) is never read again.
/// ⛔ IT EMITS NOTHING ITSELF — both of its statements are calls, and the rewrite is entry 364's.
pub fn match_and_rewrite<A: Arch>(
    send: &OpId,
    unit: &dfir::ProgramUnit<A>,
    comp: ComputeComp,
    reuse_info: &mut OperandReuse,
    is_visited: &mut BTreeMap<OpId, bool>,
    values: &mut Values,
) -> NonComputeFusion {
    let scope: &[DfirOp] = &unit.body;
    let nothing = NonComputeFusion {
        emitted: Vec::new(),
        to_erase: Vec::new(),
        outcome: FusionOutcome::NotFused,
    };

    // `:48-50` — `send_op.getSendData().getDefiningOp()`. ⛔ A `getDefiningOp()` THAT ANSWERS NULL IS
    // NOT A REFUSAL: entry 364's own gate takes `std::nullopt` as "nothing to fuse here".
    let Some(DfirOp::Dataflow(dataflow::Op::Send { data, .. })) = op_at(send, scope) else {
        return nothing;
    };
    let Some(producer) = defining_position(*data, scope) else {
        return nothing;
    };

    // `:48-50` — `VectorOperand::getOperandWithPrecision(dcc_ext_ctx_, …, comp_,
    // is_precision_converted)` at the declaration's `traverse_upwards = true` (entry 304).
    let from = VectorOperand::with_precision::<A>(&producer, comp, true, scope, &mut |op, comp| {
        VectorOperand::operand::<A>(op, comp, true, scope)
    });

    // `:53-55` — `patternAgnosticFuseNonComputeOpsHelper(send_op, from, rewriter,
    // is_precision_converted_global)`, and `return success();` under it: a `failure()` propagates as
    // itself, so the answer is handed straight back.
    pattern_agnostic_fuse_non_compute_ops_helper::<A>(
        send,
        from.operand.as_ref(),
        // ⛔ `this->unit_` AS A POSITION IS THE EMPTY PATH. The scope here IS the unit's body, so
        // every position in it is inside the unit — which is the only question entry 341 asks of it,
        // and a constant hoisted ABOVE the unit is not in this scope to be asked about at all.
        &OpId::at(&[]),
        comp,
        reuse_info,
        is_visited,
        from.precision_converted,
        scope,
        values,
    )
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 378/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e378_runOnOperation
///
/// `VectorChainToSentientPESFPLoweringPass::runOnOperation` (`VectorChainToSentientPESFP.cpp:1374`)
/// — for every PE or SFP unit, and in the order the reference calls *"Order of these operations is
/// important!"* (`:1387`): redefine the module's constant vectors, build the reuse analysis, reset
/// any existing sentient FMAs, fuse the non-compute ops, fuse the compute ops, then lower the
/// dangling non-compute ops and validate.
///
/// ⛔ `redefineConstantVectors(module_op)` TAKES THE WHOLE MODULE AND SITS INSIDE THE WALK (`:1385`)
/// — it is redone once per PE/SFP unit, not once per pass, and it is the ONE step here that is ported.
/// ⛔ THE GATE IS THE FIRST UNIT HANDLE'S TYPE (`:1379-1383`), and a unit that is neither PE nor SFP
/// is left untouched by this pass — the PT pass (entry 379) is the one that claims it.
/// ⛔ THE REUSE ANALYSIS IS SHARED BY ALL FOUR STEPS BELOW IT, so an `OperandReuse::default()` here
/// would hand the fusions an empty reuse map: every operand would look unreused and the lowering
/// would allocate a fresh register for values the reference reuses.
pub fn run_on_operation<A: Arch>(program: &mut dfir::Program<A>, values: &mut Values) {
    // `:1378` — the walk's order over the module's units. Read first because the steps below take the
    // whole module.
    let comps: Vec<DfirUnit> = program.units.iter().map(|unit| unit.on.kind()).collect();
    for comp in comps {
        // `:1383` — `is_any_of(unit_comp, PE, SFP)`.
        if !matches!(comp, DfirUnit::Pe | DfirUnit::Sfp) {
            continue;
        }

        // `:1385` — entry 067.
        redefine_constant_vectors(program, values);

        // `:1388` — `OperandReuse reuse_info(unit_op)`, which is what `:1389`, `:1391`, `:1393` and
        // `:1396`-`:1398` all read.
        todo!(
            "e227_OperandReuse is unported, so e066_resetSentientFMAsIfExists, \
             e366_fuseNonComputeOps, e076_fuseComputeOps, e344_lowerDanglingNonComputeOpsPESFP and \
             e228_validateLoweringAndSetMissingParameters cannot run on the {:?} unit",
            comp
        );
    }
}

/// Replaces: e280_cleanup
///
/// **280/384** `ComputeOpPatternBase<OpTy>::cleanup` —
/// `VectorChainToSentientPESFP.cpp:1056` (7L).
///
/// DELETE WHAT A LOWERED COMPUTE CONSUMED — the destination operands, then the compute itself, then
/// the source operands, in that order.
///
/// ⛔ ONE `erased_list` SPANS ALL THREE PHASES (`:1060`), which is why the `_recording` overloads
/// exist: an op reached from both the `to` and the `from` side is claimed once, not twice.
/// ⛔ AND NOTHING IS REMOVED UNTIL THE END, DESCENDING — an [`OpId`] is a POSITION, so erasing the
/// `to` side first would renumber the `op` and `from` positions the next two phases name.
pub fn cleanup(
    op: &OpId,
    to_operands: &[Option<VectorOperand>],
    from_operands: &[Option<VectorOperand>],
    scope: &mut Vec<DfirOp>,
) {
    // `std::vector<mlir::Operation *> erased_list;`
    let mut erased_list: Vec<OpId> = Vec::new();

    // `VectorOperand::eraseOperands(op_info.to_operands_, rewriter, erased_list);`
    erase_operands_recording(to_operands, scope, &mut erased_list);
    // `VectorOperand::eraseOp(op, rewriter, erased_list);`
    erase_op_recording(op, scope, &mut erased_list);
    // `VectorOperand::eraseOperands(op_info.from_operands_, rewriter, erased_list);`
    erase_operands_recording(from_operands, scope, &mut erased_list);

    erased_list.sort_unstable();
    erased_list.dedup();
    for position in erased_list.iter().rev() {
        remove_at(position.path(), scope);
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 344/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// ONE DUMMY MAC AND WHERE IT GOES — `OpBuilder builder(op)` (`VectorChainToSentientPESFP.cpp:1296`)
/// inserts immediately BEFORE the dangling op, which is the position recorded here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DummyMac {
    /// The dangling receive or load this stands in for.
    pub at: OpId,
    /// The `sentient.scalar_constant` mask then the `sentient.vector_mac` that reads it.
    pub ops: Vec<sen::Op>,
}

/// WHAT THE DANGLING LOWERING DID — `mlir::LogicalResult` together with the emission it made getting
/// there, because the MACs land in the sentient island while the erasures happen in the dataflow one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DanglingLowering {
    /// One per unabsorbed receive or load, in walk order.
    pub macs: Vec<DummyMac>,
    /// `result` — ⛔ THE MACS AND THE ERASURES STAND EVEN ON A FAILURE: the walk interrupts *after*
    /// them and `tobe_deleted` is drained unconditionally (`:1365-1367`).
    pub outcome: DanglingOutcome,
}

/// WHY THE DANGLING LOWERING STOPPED, kept apart because each names a different defect in the
/// program that reached it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DanglingOutcome {
    /// `result = success()` (`:1278`), never reassigned.
    Success,
    /// `emitError("There is still a receive or load with uses, that shouldn't happen")` (`:1283`).
    UsedLoad(OpId),
    /// `emitError("Dangling non-compute op has no use\n")` (`:1344`) — ⛔ IT NAMES `from.op_`, the
    /// origin, not the dangling load the walk is standing on.
    NoAbsorption(OpId),
    /// `emitError("There is still a create_affine_mask with uses - that shouldn't happen")` (`:1354`).
    UsedMask(OpId),
    /// ⛔ THE REFERENCE'S OWN ABORTS, WHICH ARE NOT `emitError`s: `getOperand(…).value()` (`:1288`),
    /// `symbolizeSentientPrecision`/`ComputePort(…).value()` (`:1315-1339`) and
    /// `llvm_unreachable("unexpected number of elements …")` (`Utils.cpp:450-451`). A stop either way.
    Unrepresentable(OpId),
}

/// WHICH FOLD MODE A COMPUTE ON `comp` CARRIES — `getSentientFoldModeAttrForOperation`
/// (`dcc/src/Dialect/Sentient/Utils.cpp:427`), which is not one of the 384 and has no other reader
/// in this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FoldModeAttr {
    /// `nullptr` — reached only when `comp` is the PT, which folds nothing.
    Absent,
    /// The attribute a PE or SFP compute carries.
    Present(sentient::FoldMode),
    /// `llvm_unreachable("unexpected number of elements for given compute precision")` (`:450-451`), and
    /// the `DT_CHECK(vector_type.has_value())` above it (`:433`).
    Unsupported,
}

/// `getSentientFoldModeAttrForOperation(op, comp, sen1p5_receive_from_pt)`
/// (`dcc/src/Dialect/Sentient/Utils.cpp:427`) in full.
///
/// ⛔ THE THIRD ARGUMENT IS WHAT MAKES THE 24-BIT REMAP REACHABLE, and it has exactly one caller
/// that passes `true`: entry 364 for a PE reading the PT FIFO on sen1p5
/// (`VectorChainToSentientPESFP.cpp:198`, `:206`); every other takes the `false` default. ⛔ AND
/// THE 16 -> 24 REWRITE RUNS **FIRST** (`dcc/src/Dialect/Sentient/Utils.cpp:436`), so a 16-bit
/// element ends at 32 through the `24 -> 32` arm — a 128-lane vector folds `fold_A` at 16, not 32.
pub(super) fn fold_mode_attr_for_operation(
    op: &DfirOp,
    comp: ComputeComp,
    sen1p5_receive_from_pt: bool,
) -> FoldModeAttr {
    // `if (is_any_of(comp, PE, SFP))`, whose `else` leaves `fold_mode` at `none`.
    if !matches!(comp, ComputeComp::Pe | ComputeComp::Sfp) {
        return FoldModeAttr::Absent;
    }
    // `auto vector_type = getVectorType(op); DT_CHECK(vector_type.has_value());`
    let Some(ty) = vector_type_of(op) else {
        return FoldModeAttr::Unsupported;
    };
    // `unsigned bitwidth = vector_type.value().getElementTypeBitWidth();` then
    // `if (comp == PE && sen1p5_receive_from_pt && bitwidth == 16) bitwidth = 24;` — ⛔ BEFORE the
    // pair below, which is what turns it into 32.
    let bitwidth = match ty.elem.bits() {
        16 if comp == ComputeComp::Pe && sen1p5_receive_from_pt => 24,
        bits => bits,
    };
    // `if (bitwidth == 80) bitwidth = 8; else if (bitwidth == 24) bitwidth = 32;`
    let bitwidth = match bitwidth {
        80 => 8,
        24 => 32,
        bits => bits,
    };
    // `int num_elements = vector_type.value().getNumElements();`
    let span = u64::from(bitwidth) * ty.len;
    if span <= 1024 {
        FoldModeAttr::Present(sentient::FoldMode::FoldA)
    } else if span == 2048 {
        FoldModeAttr::Present(sentient::FoldMode::FoldAbBoth)
    } else {
        FoldModeAttr::Unsupported
    }
}

/// `WalkOrder::PreOrder` OVER A UNIT BODY WITH EACH OP'S POSITION — the flattened numbering
/// [`OpId`] uses, and the first `Some` is `WalkResult::interrupt()`.
pub(super) fn walk_positions<T>(
    scope: &[DfirOp],
    prefix: &[u32],
    base: u32,
    found: &mut impl FnMut(&DfirOp, OpId) -> Option<T>,
) -> Option<T> {
    for (ordinal, op) in scope.iter().enumerate() {
        let mut path: Vec<u32> = prefix.to_vec();
        path.push(base + ordinal as u32);
        if let Some(hit) = found(op, OpId::at(&path)) {
            return Some(hit);
        }
        let mut child = 0u32;
        for region in regions(op) {
            if let Some(hit) = walk_positions(region, &path, child, found) {
                return Some(hit);
            }
            child += region.len() as u32;
        }
    }
    None
}

/// THE MASK CONSTANT AND THE MAC THAT STAND IN FOR ONE UNABSORBED RECEIVE OR LOAD — `:1291-1341`,
/// and [`None`] wherever the reference calls `.value()` on an empty optional.
fn dummy_mac(
    op: &DfirOp,
    at: &OpId,
    from: &VectorOperand,
    origin: Val,
    comp: ComputeComp,
    reuse: &OperandReuse,
    values: &mut Values,
) -> Option<DummyMac> {
    // `std::string input_precision = getInputPrecisionFromOperand(from);` — entry 060.
    let input_precision = input_precision_from_operand(from)?;
    // `compute_precision = from.on_the_fly_conv_precision_;`
    // `if (compute_precision != "fp32") compute_precision = "fp16";`
    let compute_precision = if from.on_the_fly_conv_precision == Some(sentient::Precision::Fp32) {
        sentient::Precision::Fp32
    } else {
        sentient::Precision::Fp16
    };
    // `getSentientFoldModeAttrForOperation(op, comp)` — the two-argument overload, i.e.
    // `sen1p5_receive_from_pt` at its `false` default.
    let fold_mode = match fold_mode_attr_for_operation(op, comp, false) {
        FoldModeAttr::Absent => None,
        FoldModeAttr::Present(mode) => Some(mode),
        FoldModeAttr::Unsupported => return None,
    };
    // `symbolizeSentientComputePort(from.getName()).value()`
    let port = from.name()?;
    // `reuse_info.getId(from.op_).value()` — ⛔ `None` IS THE `.td`'s -1, which is exactly what an
    // unassigned id answers, so the two slots the reference passes -1 outright agree with it.
    let id = match reuse.id(origin) {
        DataId::Unassigned => None,
        assigned => Some(assigned.attribute()),
    };

    // `sentient::ConstantOp::create(builder, op->getLoc(), builder.getIndexType(), 0)`
    let mask = values.mint();
    let mask_const = sen::Op::Sentient(sentient::Op::ScalarConstant {
        value: 0,
        result: mask,
        reg_locale: sentient::RegType::Imm,
        ty: ScalarTy::Index,
        is_symbol: false,
    });

    // `if (from.getName() == "west" && isa<dataflow::ReceiveOp>(op))`
    let west_receive = port == sentient::Port::West
        && matches!(op, DfirOp::Dataflow(dataflow::Op::Receive { .. }));
    // ⛔ THE PORTS AND THE ID'S SLOT MOVE TOGETHER: `one`/`zero`/port with the id on **C**, against
    // port/port/port with the id on **A** — and only the second sets `DataTransferOnly`.
    let ((a, b, c), (id_a, id_b, id_c)) = if west_receive {
        (
            (sentient::Port::One, sentient::Port::Zero, port),
            (None, None, id),
        )
    } else {
        ((port, port, port), (id, None, None))
    };
    let operand = |port: sentient::Port, data_id: Option<i32>| sentient::Operand {
        precision: input_precision,
        data_id,
        ..sentient::Operand::from(port)
    };

    let mac = sen::Op::Sentient(sentient::Op::VectorMac {
        mask: Some(mask),
        // `ValueRange(pointers)` over `ArrayRef<Value> pointers = {}` — ⛔ ALWAYS EMPTY on this path;
        // the PT pass (entry 346) is the one that threads an xrf pointer through here.
        xrf_write_ptr: None,
        xrf_read_ptr: None,
        // `TypeRange()` — a dummy MAC binds nothing.
        results: Vec::new(),
        op_a: operand(a, id_a),
        op_b: operand(b, id_b),
        op_c: operand(c, id_c),
        // `ArrayAttr::get(context, {})` for `ResultForwarding`, and
        // *"Set result precision to `none` to indicate dummy MAC"* (`:1306-1307`).
        result: sentient::ResultPorts {
            precision: sentient::Precision::None,
            ..sentient::ResultPorts::default()
        },
        // Not passed to `MacOp::create`, so the `.td` defaults stand.
        mode: sentient::FmaMode::FusedMulAdd,
        compute_precision,
        fold_mode,
        unroll_factor: sentient::UnrollFactor::X1,
        xrf_read_incr: 0,
        xrf_write_incr: 0,
        // `mac_op->setAttr("DataTransferOnly", builder.getBoolAttr(true));` (`:1341`).
        data_transfer_only: !west_receive,
        is_data_weight: None,
        // `auto op_dbg_name = dataflow::getDbgNameAttr(op);`
        dbg_name: dbg_name(op).map(str::to_owned),
    });

    Some(DummyMac {
        at: at.clone(),
        ops: vec![mask_const, mac],
    })
}

/// Replaces: e344_lowerDanglingNonComputeOpsPESFP
///
/// **344/384** `lowerDanglingNonComputeOpsPESFP` — `VectorChainToSentientPESFP.cpp:1274` (93L):
/// every receive or load left unabsorbed becomes a dummy MAC, and every leftover mask is deleted.
///
/// ⛔ A DUMMY MAC IS ONE `ResultPrecision = none` (`:1307`) AWAY FROM A REAL ONE, and only the `west`
/// receive reads `one`/`zero`/its port with the data id on **C** — every other arm reads its port
/// THREE TIMES, puts the id on **A** and sets `DataTransferOnly = true` (`:1310-1341`).
pub fn lower_dangling_non_compute_ops_pesfp<A: Arch>(
    unit: &mut dfir::ProgramUnit<A>,
    comp: ComputeComp,
    reuse: &OperandReuse,
    values: &mut Values,
) -> DanglingLowering {
    // `llvm::SmallVector<mlir::Operation *, 4> tobe_deleted;`
    let mut tobe_deleted: Vec<OpId> = Vec::new();
    // `mlir::LogicalResult result = success();`
    let mut lowering = DanglingLowering {
        macs: Vec::new(),
        outcome: DanglingOutcome::Success,
    };

    {
        let scope: &[DfirOp] = &unit.body;
        // `unit.walk<WalkOrder::PreOrder>([&](mlir::Operation *op) { … });`
        walk_positions(scope, &[], 0, &mut |op, at| {
            // `if (isa<dataflow::ReceiveOp, vector::LoadOp, agen::VectorLoadOp>(op))`
            if matches!(
                op,
                DfirOp::Dataflow(dataflow::Op::Receive { .. })
                    | DfirOp::Vector(vector::Op::Load { .. })
                    | DfirOp::Agen(agen::Op::VectorLoad { .. })
            ) {
                // `if (!op->use_empty())`
                if !use_positions(&at, scope).is_empty() {
                    lowering.outcome = DanglingOutcome::UsedLoad(at);
                    return Some(());
                }
                // `auto from = VectorOperand::getOperand(dcc_ext_ctx, op, comp).value();`
                let Some(from) = VectorOperand::operand::<A>(&at, comp, true, scope) else {
                    lowering.outcome = DanglingOutcome::Unrepresentable(at);
                    return Some(());
                };
                // `auto absorption_flag = reuse_info.getAbsorbtionFlag(from.op_);`, which this island
                // keys by the origin's VALUE — see [`origin_val`].
                let keyed = origin_val(&from.op, scope)
                    .and_then(|val| reuse.absorbtion_flag(val).map(|flag| (val, flag)));
                match keyed {
                    // `if (absorption_flag.has_value() && !absorption_flag.value())` — *"it has been
                    // used but not absorbed by some other op that is already lowered"*.
                    Some((origin, false)) => {
                        match dummy_mac(op, &at, &from, origin, comp, reuse, values) {
                            Some(mac) => lowering.macs.push(mac),
                            None => {
                                lowering.outcome = DanglingOutcome::Unrepresentable(at);
                                return Some(());
                            }
                        }
                    }
                    // `else if (!absorption_flag.has_value())`
                    None => {
                        lowering.outcome = DanglingOutcome::NoAbsorption(from.op.clone());
                        return Some(());
                    }
                    // Absorbed already: nothing is emitted, and the op is deleted all the same.
                    Some((_, true)) => {}
                }
                // `tobe_deleted.push_back(op);`
                tobe_deleted.push(at);
            } else if matches!(op, DfirOp::VectorChain(vc::Op::CreateAffineMask { .. })) {
                // *"All CreateAffineMaskOps should be connected to other operations that were already
                // lowered."*
                if !use_positions(&at, scope).is_empty() {
                    lowering.outcome = DanglingOutcome::UsedMask(at);
                    return Some(());
                }
                tobe_deleted.push(at);
            }
            // `return WalkResult::advance();`
            None
        });
    }

    // `for (auto op : tobe_deleted) VectorOperand::eraseOp(op);` — ⛔ DESCENDING, because an [`OpId`]
    // is a POSITION and removing `[3]` renumbers `[4]`; an `Operation *` needs no such ordering.
    tobe_deleted.sort_unstable();
    tobe_deleted.dedup();
    for position in tobe_deleted.iter().rev() {
        erase_op(position, &mut unit.body);
    }

    // `return result;`
    lowering
}

#[cfg(test)]
mod unit_tests {
    use super::{
        COMPUTE_OPS_PATTERNS, ComputeComp, ComputePattern, DanglingOutcome, FilledOpInfo,
        FusionOutcome, InsertAt, LEGAL_DIALECTS, Legality, OpId, Unlowered, VectorOperand, cleanup,
        compute_ops_to_fuse, fill_op_info, fuse_compute_ops, installed_pattern, legality,
        lower_dangling_non_compute_ops_pesfp, match_and_rewrite, non_compute_ops_to_fuse,
        run_on_operation,
    };
    use crate::arch::Target;
    use crate::bridges::dataflow_ir_to_sentient::vc_operand_reuse::OperandReuse;
    use crate::bridges::dataflow_ir_to_sentient::vc_vector_operands::{
        OperandValue, VectorOperandType,
    };
    use crate::islands::dataflow_ir::Values;
    use crate::islands::dataflow_ir::dialects::dataflow;
    use crate::islands::dataflow_ir::dialects::vectorchain as vc;
    use crate::islands::dataflow_ir::dialects::{Op as DfirOp, Val, affine, arith};
    use crate::islands::dataflow_ir::link::{Link, Lxlu, Lxsu, Pe, Sfp};
    use crate::islands::dataflow_ir::ty::{AffineExpr, AffineMap, ElemType, IntegerSet, Vector};
    use crate::islands::dataflow_ir::{self as dfir, ProgramUnit, Units};
    use crate::islands::sentient::dialects::Op as SenOp;
    use crate::islands::sentient::dialects::sentient as sen;
    use crate::units::{Core, Corelet, DfirUnit, Residency};
    use std::collections::BTreeMap;

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

    /// One program holding one SFP unit — what this pass walks.
    fn program_of(body: Vec<DfirOp>) -> dfir::Program<Target> {
        use crate::generated::OpFunc;
        use crate::islands::dataflow_ir::{Grid, GroupId, OpIndex, ProgramName, ProgramUnits};
        dfir::Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            grid: Grid::single(),
            preamble: Vec::new(),
            units: ProgramUnits::of(unit_holding(body), Vec::new()),
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
                symbols: 0,
                constraints: Vec::new(),
            },
            access_function_ca: map(),
            access_function_a: map(),
            iteration_space_cb: IntegerSet {
                dims: 1,
                symbols: 0,
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
            is_symbol: false,
        }));
        ops.push(DfirOp::VectorChain(vc::Op::Shuffle {
            result: Val(0),
            input: Val(1),
            variable: Vec::new(),
            pad: Vec::new(),
            mask: None,
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
            dbg_name: None,
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
            variable: Vec::new(),
            pad: Vec::new(),
            mask: None,
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

    /// 🎯 377/384 AND 364/384 — THE VENDOR'S OWN CASE, `opA-forwarding.mlir:73-74`: an `lxlu`
    /// receive feeding a send to the `sfp` on a PE unit becomes ONE `sentient.vector_binary` OR-ing
    /// `lx` with `zero` and forwarding `opA` to `sfp` (`:16-17` of the same file's expectations).
    /// ⛔ THE RESULT FORWARDING IS EMPTY AND `opA`'S IS NOT — a LOGICAL fusion may not forward
    /// src0/src2 through `tgtrf` (`:237`), so a non-register destination lands on **A**.
    /// ⛔ AND `opBDataID` IS UNASSIGNED HERE. The vendor prints `6`, which entry 228 fills in after
    /// both fusions (`VectorChainHelper.cpp:483-533`); this pass emits the `.td`'s -1.
    #[test]
    fn a_receive_and_its_send_fuse_into_one_or_with_zero() {
        const F16X64: Vector = Vector {
            len: 64,
            elem: ElemType::F16,
        };
        let corelet0 = || Residency::Corelet {
            core: Core::checked(0).expect("the arch has core 0"),
            corelet: Corelet::checked(0).expect("the arch has corelet 0"),
        };
        let mut vals = Values::default();
        let (lxlu, pe, sfp_unit, data) = (vals.mint(), vals.mint(), vals.mint(), vals.mint());
        // ⛔ BOTH ENDS NEED THEIR `get_unit` IN THE BODY: an operand is resolved by walking a link
        // end back to the op that binds it, and `lx` comes from the SOURCE's kind, `sfp` from the
        // DESTINATION's.
        let (_, from) = Link::<Lxlu, Pe>::between(lxlu, pe).ends();
        let (to, _) = Link::<Pe, Sfp>::between(pe, sfp_unit).ends();
        let unit = ProgramUnit::<Target> {
            on: Units::one(DfirUnit::Pe, pe),
            precision: None,
            body: vec![
                DfirOp::Dataflow(dataflow::Op::GetUnit {
                    result: lxlu,
                    residency: corelet0(),
                    unit: DfirUnit::Lxlu,
                    num_folds: None,
                    reg_locale: None,
                }),
                DfirOp::Dataflow(dataflow::Op::GetUnit {
                    result: sfp_unit,
                    residency: corelet0(),
                    unit: DfirUnit::Sfp,
                    num_folds: None,
                    reg_locale: None,
                }),
                DfirOp::Dataflow(dataflow::Op::Receive {
                    result: data,
                    from,
                    ty: F16X64,
                }),
                DfirOp::Dataflow(dataflow::Op::Send {
                    to,
                    data,
                    ty: F16X64,
                    dir: None,
                }),
            ],
            arch: core::marker::PhantomData,
        };

        let mut reuse = OperandReuse::default();
        let mut is_visited = BTreeMap::new();
        let fusion = match_and_rewrite(
            &OpId::at(&[3]),
            &unit,
            ComputeComp::Pe,
            &mut reuse,
            &mut is_visited,
            &mut vals,
        );

        assert_eq!(fusion.outcome, FusionOutcome::Fused);
        let [group] = fusion.emitted.as_slice() else {
            panic!("one insertion point, not {:?}", fusion.emitted)
        };
        // ⛔ IN FRONT OF THE SEND, because no destination was `sfpring` — see [`InsertAt`].
        assert_eq!(group.at, InsertAt::Before(OpId::at(&[3])));
        let [mask_const, compute] = group.ops.as_slice() else {
            panic!("a mask constant and one compute, not {:?}", group.ops)
        };
        let SenOp::Sentient(sen::Op::ScalarConstant {
            value: 0,
            result: mask,
            ..
        }) = mask_const
        else {
            panic!("a mask constant, not {mask_const:?}")
        };
        let SenOp::Sentient(sen::Op::VectorBinary {
            mask: masked_by,
            op_a,
            op_b,
            binary_op,
            result,
            compute_precision,
            fold_mode,
            ..
        }) = compute
        else {
            panic!("a vector_binary, not {compute:?}")
        };
        assert_eq!(masked_by, mask);
        assert_eq!(
            op_a,
            &sen::Operand {
                forwarding: vec![sen::Port::Sfp],
                precision: sen::Precision::Fp16,
                data_id: Some(0),
                ..sen::Operand::from(sen::Port::Lx)
            }
        );
        assert_eq!(
            op_b,
            &sen::Operand {
                precision: sen::Precision::Fp16,
                ..sen::Operand::from(sen::Port::Zero)
            }
        );
        assert!(result.forwarding.is_empty());
        assert_eq!(
            (binary_op, result.precision, *compute_precision, *fold_mode),
            (
                &sen::Binary::Plain(sen::BinaryOp::Or),
                sen::Precision::Fp16,
                // *"Bitwise operation so set compute precision to none."*
                sen::Precision::None,
                // 16 bits x 64 lanes = 1024, the `<= 1024` arm.
                Some(sen::FoldMode::FoldA)
            )
        );
        // ⛔ THE RECEIVE GOES TOO, and only because nothing dangles after the fusion (`:310`) — the
        // send is its one user and the fused compute has taken its place.
        assert_eq!(fusion.to_erase, vec![OpId::at(&[2]), OpId::at(&[3])]);
    }

    /// 🎯 378/384 — THE PASS CLAIMS THE SFP AND STOPS AT THE SHARED REUSE ANALYSIS. `redefine_constant_vectors`
    /// (entry 067) runs first, inside the walk, before anything asks for the reuse map (`:1385`-`:1388`).
    #[test]
    #[should_panic(expected = "e227_OperandReuse")]
    fn a_pe_or_sfp_unit_is_claimed_by_this_pass() {
        let mut program = program_of(vec![DfirOp::Affine(affine::Op::Yield {
            operands: Vec::new(),
        })]);
        run_on_operation(&mut program, &mut Values::default());
    }

    /// 🎯⛔ 280/384 — THE WHOLE LOWERED CHAIN GOES, AND ONLY BECAUSE THE THREE PHASES SHARE ONE LIST.
    /// The `to` send at `[3]` is claimed first, which leaves the compute at `[2]` unread, which leaves
    /// the cast at `[1]` unread, which lets the `from` receive at `[0]` go. A phase that could still
    /// see a claimed op's uses would stop at `[2]` and leave the source chain in the unit.
    #[test]
    fn the_to_side_being_claimed_first_is_what_frees_the_from_side() {
        let (to, from) = Link::<Sfp, Lxsu>::between(Val(7), Val(8)).ends();
        let mut scope = vec![
            DfirOp::Dataflow(dataflow::Op::Receive {
                result: Val(0),
                from,
                ty: V,
            }),
            DfirOp::VectorChain(vc::Op::Cast {
                result: Val(1),
                input: Val(0),
                input_ty: V,
                ty: V,
            }),
            DfirOp::VectorChain(vc::Op::FastExp {
                result: Val(2),
                input: Val(1),
                input_ty: V,
                ty: V,
            }),
            DfirOp::Dataflow(dataflow::Op::Send {
                to,
                data: Val(2),
                ty: V,
                dir: None,
            }),
        ];
        let at = |path: &[u32]| {
            Some(VectorOperand::new(
                VectorOperandType::Link,
                OperandValue::Port(sen::Port::North),
                OpId::at(path),
            ))
        };
        cleanup(&OpId::at(&[2]), &[at(&[3])], &[at(&[0])], &mut scope);
        assert_eq!(scope, Vec::new());
    }

    /// 🎯 344/384 — THE VENDOR'S OWN CASE, `sfp-to-sfp-ring.mlir:110-114`: three `lx` receives feed
    /// one `multiply_and_accumulate`, whose fusion absorbs the third and latches the other two.
    /// Those two come back as dummy MACs, printed at `sfp-to-sfp-ring.mlir:24-26` with
    /// `DataTransferOnly = true`, `ResultPrecision = none` and `fold_A` for 64 f16 elements.
    /// ⛔ THE DATA ID SITS ON **A** ALONE; `opB`/`opC` keep the `.td`'s -1 until e228 fills them.
    #[test]
    fn the_two_latched_receives_come_back_as_data_transfer_only_macs() {
        const F16X64: Vector = Vector {
            len: 64,
            elem: ElemType::F16,
        };
        let mut vals = Values::default();
        let (_, from) = Link::<Sfp, Lxsu>::between(vals.mint(), vals.mint()).ends();
        let data: Vec<Val> = (0..3).map(|_| vals.mint()).collect();
        // ⛔ THE SOURCE UNIT HAS TO BE IN THE BODY. `VectorOperand::getOperand` resolves a receive's
        // link end back to the `dataflow.get_unit` that binds it, and with no such op every receive
        // is `Unrepresentable` before any absorption flag is read — the receives stay at `[0..2]`, so
        // the unit goes after them.
        let mut unit = unit_holding(
            data.iter()
                .map(|result| {
                    DfirOp::Dataflow(dataflow::Op::Receive {
                        result: *result,
                        from,
                        ty: F16X64,
                    })
                })
                .chain(core::iter::once(DfirOp::Dataflow(dataflow::Op::GetUnit {
                    result: from.val(),
                    residency: Residency::Corelet {
                        core: Core::checked(0).expect("the arch has core 0"),
                        corelet: Corelet::checked(0).expect("the arch has corelet 0"),
                    },
                    unit: DfirUnit::Lxsu,
                    num_folds: None,
                    reg_locale: None,
                })))
                .collect(),
        );
        // `reuse_info` as the compute fusion leaves it: the accumulator got absorbed, the two
        // operands the MAC re-reads from the latch did not.
        let lx = |at: u32| {
            VectorOperand::new(
                VectorOperandType::Link,
                OperandValue::Port(sen::Port::Lx),
                OpId::at(&[at]),
            )
        };
        let mut reuse = OperandReuse::default();
        reuse.set_reuse_information(&OpId::at(&[4]), &mut [lx(0), lx(1), lx(2)], &unit.body);

        let lowering =
            lower_dangling_non_compute_ops_pesfp(&mut unit, ComputeComp::Sfp, &reuse, &mut vals);

        assert_eq!(lowering.outcome, DanglingOutcome::Success);
        // ⛔ ALL THREE RECEIVES GO, absorbed or not — `tobe_deleted` is pushed outside the `if`. The
        // `get_unit` is not one of the three op classes the walk deletes, so it stays.
        assert_eq!(unit.body.len(), 1);
        let [first, second] = lowering.macs.as_slice() else {
            panic!("two dummy MACs, not {:?}", lowering.macs)
        };
        assert_eq!((&first.at, &second.at), (&OpId::at(&[0]), &OpId::at(&[1])));
        let SenOp::Sentient(sen::Op::VectorMac {
            mask,
            op_a,
            op_b,
            op_c,
            result,
            compute_precision,
            fold_mode,
            data_transfer_only,
            ..
        }) = &first.ops[1]
        else {
            panic!("a mac, not {:?}", first.ops[1])
        };
        let lx_at = |data_id| sen::Operand {
            precision: sen::Precision::Fp16,
            data_id,
            ..sen::Operand::from(sen::Port::Lx)
        };
        assert_eq!(
            (op_a, op_b, op_c),
            (&lx_at(Some(0)), &lx_at(None), &lx_at(None))
        );
        assert_eq!(
            (
                result.precision,
                *compute_precision,
                *fold_mode,
                *data_transfer_only
            ),
            (
                sen::Precision::None,
                sen::Precision::Fp16,
                Some(sen::FoldMode::FoldA),
                true
            )
        );
        // `sentient.scalar_constant {value = 0 : si64} : index`, and the MAC masks with its result.
        let SenOp::Sentient(sen::Op::ScalarConstant { value, result, .. }) = &first.ops[0] else {
            panic!("a mask constant, not {:?}", first.ops[0])
        };
        assert_eq!((*value, mask), (0, &Some(*result)));
        // The second dummy MAC differs in exactly one place: `opADataID = 1`.
        let SenOp::Sentient(sen::Op::VectorMac { op_a, .. }) = &second.ops[1] else {
            panic!("a mac, not {:?}", second.ops[1])
        };
        assert_eq!(op_a, &lx_at(Some(1)));
    }

    /// 🎯 366/384 — THE INVERSION AT `:1230`. A receive whose ONLY user is a send fuses, so
    /// `is_fusion_respected` stays true and the lambda answers *illegal*; add one user that is
    /// neither a send nor a store and entry 341 clears the flag, which answers *legal* and leaves
    /// the send exactly where it is.
    #[test]
    fn a_send_fed_by_a_lone_receive_is_the_illegal_op_and_a_second_user_makes_it_legal() {
        let mut vals = Values::default();
        let sfp = vals.mint();
        let (_, from) = Link::<Lxlu, Sfp>::between(vals.mint(), sfp).ends();
        let (to, _) = Link::<Sfp, Lxsu>::between(sfp, vals.mint()).ends();
        let data = vals.mint();
        let unit = |extra: Vec<DfirOp>| {
            unit_holding(
                vec![
                    // ⛔ BOTH ENDS NEED THEIR `get_unit` IN THE BODY, or neither operand resolves —
                    // see the 344 fixture's note.
                    DfirOp::Dataflow(dataflow::Op::GetUnit {
                        result: from.val(),
                        residency: Residency::Global,
                        unit: DfirUnit::Lxlu,
                        num_folds: None,
                        reg_locale: None,
                    }),
                    DfirOp::Dataflow(dataflow::Op::GetUnit {
                        result: to.val(),
                        residency: Residency::Global,
                        unit: DfirUnit::Lxsu,
                        num_folds: None,
                        reg_locale: None,
                    }),
                    DfirOp::Dataflow(dataflow::Op::Receive {
                        result: data,
                        from,
                        ty: V,
                    }),
                    DfirOp::Dataflow(dataflow::Op::Send {
                        to,
                        data,
                        ty: V,
                        dir: None,
                    }),
                ]
                .into_iter()
                .chain(extra)
                .collect(),
            )
        };

        assert_eq!(
            non_compute_ops_to_fuse::<Target>(&unit(Vec::new()), ComputeComp::Sfp),
            vec![OpId::at(&[3])]
        );
        assert_eq!(
            non_compute_ops_to_fuse::<Target>(
                &unit(vec![DfirOp::VectorChain(vc::Op::FastExp {
                    result: vals.mint(),
                    input: data,
                    input_ty: V,
                    ty: V,
                })]),
                ComputeComp::Sfp
            ),
            Vec::new()
        );
    }

    // ── 365/384 ───────────────────────────────────────────────────────────────────────────────

    /// 🎯 365/384 — THE MASK IS THE OPERAND **PAST** THE COUNTED ONES, AND AN ABSENT SLOT CARRIES
    /// THE COMPUTE PRECISION.
    ///
    /// A masked `vectorchain.binary` has three operands where `num_operands` is two, so operand 2 is
    /// the mask: 96 live lanes of 128 masks slices 6 and 7, `{value = 192 : si64}`. The unary
    /// `vectorchain.fast_exp` beside it has ONE operand, and its B and C slots carry the COMPUTE
    /// precision — `bf16` remapped to `fp16` — where A carries the receive's own `bf16`.
    #[test]
    fn the_mask_is_the_operand_past_the_counted_ones_and_an_absent_slot_takes_the_compute_precision()
     {
        let mut vals = Values::default();
        let (_, from) = Link::<Lxlu, Sfp>::between(Val(10), Val(11)).ends();
        let (data, masked) = (Val(1), Val(2));
        const I1: Vector = Vector {
            len: 128,
            elem: ElemType::Int(1),
        };
        let scope = vec![
            DfirOp::Dataflow(dataflow::Op::GetUnit {
                result: Val(10),
                residency: Residency::Global,
                unit: DfirUnit::Lxlu,
                num_folds: None,
                reg_locale: None,
            }),
            DfirOp::Dataflow(dataflow::Op::Receive {
                result: data,
                from,
                ty: V,
            }),
            DfirOp::VectorChain(vc::Op::CreateAffineMask {
                result: masked,
                mask: vc::LaneMask::prefix_of(96, I1),
            }),
            DfirOp::VectorChain(vc::Op::Binary {
                result: Val(3),
                op1: data,
                op2: data,
                mask: Some(vc::LaneMask::prefix_of(96, I1).binds(masked)),
                binary_op: vc::BinaryOp::Add,
                op_specific_map: map(),
                operand_ty: V,
                ty: V,
            }),
            DfirOp::VectorChain(vc::Op::FastExp {
                result: Val(4),
                input: data,
                input_ty: V,
                ty: V,
            }),
        ];

        let FilledOpInfo::Filled(binary) = fill_op_info::<Target>(
            &scope[3],
            2,
            &OpId::at(&[3]),
            ComputeComp::Sfp,
            &scope,
            &mut vals,
        ) else {
            panic!("a masked binary fills")
        };
        assert_eq!(binary.mask_operand, Some(OpId::at(&[2])));
        let SenOp::Sentient(sen::Op::ScalarConstant { value, result, .. }) = &binary.mask_op else {
            panic!("a mask constant, not {:?}", binary.mask_op)
        };
        assert_eq!((*value, binary.mask_val), (192, *result));
        assert_eq!(
            (
                binary.op_a_precision,
                binary.op_b_precision,
                binary.compute_precision
            ),
            (
                Some(sen::Precision::Bf16),
                Some(sen::Precision::Bf16),
                sen::Precision::Fp16
            )
        );

        // ⛔ AND THE UNARY OP'S B AND C SLOTS ARE THE COMPUTE PRECISION — not empty, and not A's.
        let FilledOpInfo::Filled(unary) = fill_op_info::<Target>(
            &scope[4],
            1,
            &OpId::at(&[4]),
            ComputeComp::Sfp,
            &scope,
            &mut vals,
        ) else {
            panic!("a unary op fills")
        };
        assert_eq!(
            (
                unary.op_a_precision,
                unary.op_b_precision,
                unary.op_c_precision
            ),
            (
                Some(sen::Precision::Bf16),
                Some(sen::Precision::Fp16),
                Some(sen::Precision::Fp16)
            )
        );
        // Nothing reads it, so the result precision is the substituted compute precision and the
        // mask is the unmasked arm's own `sentient.constant 0`.
        assert_eq!(unary.mask_operand, None);
        assert_eq!(unary.result_precision, Some(sen::Precision::Fp16));
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 365/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// EVERYTHING THE SIXTEEN COMPUTE PATTERNS READ OFF ONE `vectorchain` OP — `OpInfo`
/// (`VectorChainToSentientPESFP.cpp`'s pass class), filled in one pass by [`fill_op_info`].
///
/// ⛔ THE THREE OPERAND SLOTS ARE NOT A LIST. `opA_`/`opB_`/`opC_` are the `sentient.compute`'s three
/// named ports, and each carries its own forwarding list and its own precision — a `Vec` of triples
/// would let a pattern read B's precision into C's slot and still typecheck.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpInfo {
    /// `from_operands_` — one per source operand, in operand order. ⛔ An ABSENT operand is KEPT, as
    /// entry 342 records for the `to` side.
    pub from_operands: Vec<Option<VectorOperand>>,
    /// `mask_operand_` — the op that produced the mask, as a position. See [`fill_op_info`]'s note on
    /// the null pointer the reference can store here.
    pub mask_operand: Option<OpId>,
    /// `opA_forwarding_`, `opB_forwarding_`, `opC_forwarding_` — entry 340, once per present operand.
    pub op_a_forwarding: Vec<sentient::Port>,
    /// See [`Self::op_a_forwarding`].
    pub op_b_forwarding: Vec<sentient::Port>,
    /// See [`Self::op_a_forwarding`].
    pub op_c_forwarding: Vec<sentient::Port>,
    /// `mask_val_` — the value the `sentient.vector_mac` reads as its mask.
    pub mask_val: Val,
    /// The op that binds [`Self::mask_val`]: entry 229's `sentient.scalar_constant`, or the literal
    /// `sentient.constant 0` the unmasked arm creates.
    pub mask_op: sen::Op,
    /// `to_operands_` — entry 342's, one per user.
    pub to_operands: Vec<Option<VectorOperand>>,
    /// `result_forwarding_` — entry 342's.
    pub result_forwarding: Vec<sentient::Port>,
    /// `logical_result_forwarding_` — the `istate` destination entry 342 splits off.
    pub logical_result_forwarding: Option<sentient::Port>,
    /// The `sentient.set_send_dst` ops entry 342 emits for an `sfpring` destination, which the caller
    /// positions after the send.
    pub set_send_dst: Vec<sen::Op>,
    /// `result_precision_` — entry 061's, with the compute precision substituted where there is no
    /// result forwarding and no precision.
    pub result_precision: Option<sentient::Precision>,
    /// `compute_precision_` — entry 062's.
    pub compute_precision: sentient::Precision,
    /// `fold_mode_attr_` — [`fold_mode_attr_for_operation`].
    pub fold_mode_attr: FoldModeAttr,
    /// `opA_precision_` — entry 060's on operand 0.
    pub op_a_precision: Option<sentient::Precision>,
    /// `opB_precision_` — entry 060's on operand 1, or the COMPUTE precision when there is no
    /// operand 1.
    pub op_b_precision: Option<sentient::Precision>,
    /// `opC_precision_` — entry 060's on operand 2, or the COMPUTE precision when there is no
    /// operand 2.
    pub op_c_precision: Option<sentient::Precision>,
    /// `op_dbg_name_` — `dataflow::getDbgNameAttr(op)`.
    pub op_dbg_name: Option<String>,
}

/// WHAT `fillOpInfo` ANSWERS — `mlir::LogicalResult`, with the filled record on the success side.
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub enum FilledOpInfo {
    /// `return success();`
    Filled(OpInfo),
    /// `if (!op_info.mask_val_.has_value()) return failure();` — the mask op carries no constant
    /// affine set, which is entry 229's only refusal.
    MaskHasNoConstantValue(OpId),
    /// The op has fewer than one operand, so `from_operands_[0]` is out of range — the reference
    /// indexes it unguarded (`:1099`).
    NoOperands,
}

/// Replaces: e365_fillOpInfo
///
/// **365/384** `VectorChainToSentientPESFPLoweringPass::fillOpInfo` —
/// `VectorChainToSentientPESFP.cpp:1070` (83L): read one compute's operands, mask, forwarding and
/// precisions into the record its lowering pattern emits from.
///
/// ⛔ THE MASK IS THE OPERAND **PAST** THE COUNTED ONES (`:1092`), which is why `num_operands` is a
/// parameter and not `op->getNumOperands()`: a masked binary has three operands and `num_operands` is
/// two. ⛔ A `vectorchain.shuffle` is the exception and reads `getMask()` instead (`:1082-1090`),
/// because its two VARIADIC operand groups sit between the input and the mask.
/// ⛔ `opB`/`opC` PRECISION FALL BACK TO THE **COMPUTE** PRECISION, NOT TO EMPTY (`:1144-1153`) — a
/// unary op's B and C slots carry the compute precision on the wire.
/// ⚠️ A MASK BOUND BY A REGION ARGUMENT: the reference stores a NULL `Operation*` in a live
/// `std::optional` and dereferences it in entry 229; `defining_position` answers [`None`], which takes
/// the unmasked arm and its `sentient.constant 0`.
pub fn fill_op_info<A: Arch>(
    op: &DfirOp,
    num_operands: usize,
    at: &OpId,
    comp: ComputeComp,
    scope: &[DfirOp],
    values: &mut Values,
) -> FilledOpInfo {
    // `:1076-1080` — `for (int i = 0; i < num_operands; i++)`, each through entry 320.
    let reads = op_operands(op);
    let mut from_operands: Vec<Option<VectorOperand>> = Vec::new();
    for read in reads.iter().take(num_operands) {
        let operand_op = defining_position(*read, scope);
        from_operands
            .push(operand_op.and_then(|operand_op| {
                VectorOperand::operand::<A>(&operand_op, comp, true, scope)
            }));
    }

    // `:1082-1094` — the mask operand, and the two ways of naming it.
    let mask_operand = match op {
        // *"Shuffle operations contain two variadic operands before the mask operand."*
        DfirOp::VectorChain(vc::Op::Shuffle { mask, .. }) => {
            mask.and_then(|mask| defining_position(mask.val(), scope))
        }
        // `else if (op->getNumOperands() == num_operands + 1)`.
        _ if reads.len() == num_operands + 1 => defining_position(reads[num_operands], scope),
        _ => None,
    };

    // `:1098-1112` — entry 340 on operand 0, then on 1 and 2 only if the list is that long. ⛔ THE
    // FIRST CALL IS UNGUARDED, so an op with no operands indexes out of range there.
    if from_operands.is_empty() {
        return FilledOpInfo::NoOperands;
    }
    let mut op_a_forwarding: Vec<sentient::Port> = Vec::new();
    let mut op_b_forwarding: Vec<sentient::Port> = Vec::new();
    let mut op_c_forwarding: Vec<sentient::Port> = Vec::new();
    analyze_and_fill_operand_forwarding::<A>(
        comp,
        from_operands[0].as_ref(),
        scope,
        &mut op_a_forwarding,
    );
    if from_operands.len() > 1 {
        analyze_and_fill_operand_forwarding::<A>(
            comp,
            from_operands[1].as_ref(),
            scope,
            &mut op_b_forwarding,
        );
    }
    if from_operands.len() > 2 {
        analyze_and_fill_operand_forwarding::<A>(
            comp,
            from_operands[2].as_ref(),
            scope,
            &mut op_c_forwarding,
        );
    }

    // `:1114-1124` — *"If no mask info is attached to the op, use the default mask, 0."*
    let mask_vc_op = mask_operand
        .as_ref()
        .and_then(|mask| op_at(mask, scope))
        .and_then(|mask| match mask {
            DfirOp::VectorChain(mask) => Some(mask),
            _ => None,
        });
    let (mask_val, mask_op) = match mask_vc_op {
        Some(mask) => match get_mask_value_for_non_pt::<A>(mask, values) {
            Some(mask) => (mask.value, mask.op),
            // `:1124` — `if (!op_info.mask_val_.has_value()) return failure();`
            None => {
                return FilledOpInfo::MaskHasNoConstantValue(at.clone());
            }
        },
        // `sentient::ConstantOp::create(rewriter, op->getLoc(), rewriter.getIndexType(), 0)`.
        None => {
            let value = values.mint();
            (
                value,
                sen::Op::Sentient(sentient::Op::ScalarConstant {
                    value: 0,
                    result: value,
                    reg_locale: sentient::RegType::Imm,
                    ty: ScalarTy::Index,
                    is_symbol: false,
                }),
            )
        }
    };

    // `:1126-1128` — entry 342.
    let mut to_operands: Vec<Option<VectorOperand>> = Vec::new();
    let mut result_forwarding: Vec<sentient::Port> = Vec::new();
    let mut logical_result_forwarding: Option<sentient::Port> = None;
    let mut set_send_dst: Vec<sen::Op> = Vec::new();
    analyze_and_fill_result_forwarding::<A>(
        at,
        comp,
        scope,
        &mut to_operands,
        &mut result_forwarding,
        &mut logical_result_forwarding,
        &mut set_send_dst,
    );

    // `:1130-1132` — entries 061 and 062.
    let mut result_precision = result_precision_from_operands(&to_operands);
    let compute_precision = compute_precision_of_op(op);
    // `:1135-1136` — *"If no result forwarding, set result_precision to default precision."*
    if result_forwarding.is_empty() && result_precision.is_none() {
        result_precision = Some(compute_precision);
    }

    FilledOpInfo::Filled(OpInfo {
        // `:1144-1153` — and the compute precision is what an absent B or C slot carries.
        op_a_precision: from_operands[0]
            .as_ref()
            .and_then(input_precision_from_operand),
        op_b_precision: match from_operands.get(1) {
            Some(operand) => operand.as_ref().and_then(input_precision_from_operand),
            None => Some(compute_precision),
        },
        op_c_precision: match from_operands.get(2) {
            Some(operand) => operand.as_ref().and_then(input_precision_from_operand),
            None => Some(compute_precision),
        },
        from_operands,
        mask_operand,
        op_a_forwarding,
        op_b_forwarding,
        op_c_forwarding,
        mask_val,
        mask_op,
        to_operands,
        result_forwarding,
        logical_result_forwarding,
        set_send_dst,
        result_precision,
        compute_precision,
        // `:1139` — `getSentientFoldModeAttrForOperation(op, comp)` at the declaration's
        // `sen1p5_receive_from_pt = false`; only entry 364 passes `true`.
        fold_mode_attr: fold_mode_attr_for_operation(op, comp, false),
        // `:1155` — `dataflow::getDbgNameAttr(op)`.
        op_dbg_name: dbg_name(op).map(str::to_owned),
    })
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 366/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHAT `fuseNonComputeOps`'s TARGET SAYS ABOUT ONE OP — `VectorChainToSentientPESFP.cpp:1178-1236`.
///
/// ⛔⛔ THERE IS NO `addIllegalOp` IN THIS FUNCTION, unlike [`legality`]'s thirteen. So
/// [`NonComputeLegality::Unnamed`] can never fail the conversion, and the ONLY behaviour in the whole
/// target is [`NonComputeLegality::Dynamic`]'s callback over three op classes.
/// ⭐ `vectorchain` IS LEGAL HERE (`:1180`) and absent from [`LEGAL_DIALECTS`] — the non-compute half
/// runs FIRST (`:1391`, `:1393`), so the computes it must not touch are declared legal outright,
/// while `agen`, `trace` and `dataflow` are enumerated op by op (`:1183-1197`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NonComputeLegality {
    /// `addLegalDialect` or one of the three `addLegalOp` lists names it.
    Legal,
    /// `addDynamicallyLegalOp<dataflow::SendOp, vector::StoreOp, agen::VectorStoreOp>` (`:1232-1234`).
    Dynamic,
    /// Named by neither list: partial conversion offers it to the patterns and leaves it alone.
    Unnamed,
}

/// THE TARGET AS ONE EXHAUSTIVE MATCH. ⛔ EXHAUSTIVE DELIBERATELY, as [`legality`] is: an op added to
/// the island later must be classified here rather than defaulted.
#[must_use]
pub fn non_compute_legality(op: &DfirOp) -> NonComputeLegality {
    match op {
        // ── the three dynamically legal ops (`:1232-1234`) ────────────────────────────────────────
        DfirOp::Dataflow(dataflow::Op::Send { .. })
        | DfirOp::Vector(vector::Op::Store { .. })
        | DfirOp::Agen(agen::Op::VectorStore { .. }) => NonComputeLegality::Dynamic,

        // ── `addLegalDialect<arith, vectorchain, sentient, memref, uniform, symbol>` (`:1179-1182`);
        //    `sentient` and `memref` have no `Op` arm of their own in this island ─────────────────
        DfirOp::Arith(_) | DfirOp::VectorChain(_) | DfirOp::Uniform(_) | DfirOp::Symbol(_) => {
            NonComputeLegality::Legal
        }

        // ── `addLegalOp<agen::YieldOp, …, agen::SetTransferMaskStateOp>` (`:1183-1189`) ───────────
        DfirOp::Agen(
            agen::Op::Yield
            | agen::Op::VectorLoad { .. }
            | agen::Op::CompositeLoad(_)
            | agen::Op::CompositeStore(_)
            | agen::Op::CompositeLoadAndStore(_)
            | agen::Op::CompositeIndirectLoadAndStore(_)
            | agen::Op::CompositeIndirectLoad(_)
            | agen::Op::CompositeIndirectStore(_)
            | agen::Op::IndirectVectorLoad { .. }
            | agen::Op::IndirectVectorStore { .. }
            | agen::Op::CompositeMemoryInterleave { .. }
            | agen::Op::SetTransferMaskState { .. },
        ) => NonComputeLegality::Legal,

        // ── `addLegalOp<dataflow::GetUnitOp, …, dataflow::OpaqueOp>` (`:1191-1197`). The four
        //    collection ops are this island's, are named by neither list, and so are `Unnamed` ────
        DfirOp::Dataflow(
            dataflow::Op::GetUnit { .. }
            | dataflow::Op::GetLocalUnit { .. }
            | dataflow::Op::ProgramUnit { .. }
            | dataflow::Op::CreateGroup { .. }
            | dataflow::Op::SyncSend { .. }
            | dataflow::Op::SyncRecv { .. }
            | dataflow::Op::ImplicitSync { .. }
            | dataflow::Op::GetLogicalMemoryView { .. }
            | dataflow::Op::Receive { .. },
        ) => NonComputeLegality::Legal,

        // ── `vector.load`, `scf`, `affine`, the symbolic agen accesses and the collection ops:
        //    unnamed, and therefore untouched ───────────────────────────────────────────────────
        DfirOp::Vector(_)
        | DfirOp::Scf(_)
        | DfirOp::Affine(_)
        | DfirOp::Agen(_)
        | DfirOp::Dataflow(_) => NonComputeLegality::Unnamed,
    }
}

/// WHICH SENDS AND STORES THE THREE PATTERNS WOULD REWRITE IN ONE UNIT, IN PREORDER — the
/// `getDynamicLoweringLegality` lambda (`:1201-1230`) asked of every [`NonComputeLegality::Dynamic`]
/// op.
///
/// ⛔⛔ THE LAMBDA'S OWN COMMENT IS INVERTED. `addDynamicallyLegalOp`'s callback answers *is this op
/// LEGAL*, so `return !is_fusion_respected` means a RESPECTED fusion is the illegal op the pattern
/// rewrites, and the trailing `return true` leaves a send whose producer is not a
/// receive/load/constant exactly where it is. Reading the comment instead of the contract inverts
/// the pass.
/// ⛔ `is_visited` IS MUTATED BY THE QUERY, through entry 341 — so the answer for one op depends on
/// which ops were asked before it. The queries are made in the same preorder as the seeding walk.
/// ⭐ SEPARATE FROM [`fuse_non_compute_ops`] SO IT IS TESTABLE: the pass can only fail on a non-empty
/// answer, and the answer is the part with content.
#[must_use]
pub fn non_compute_ops_to_fuse<A: Arch>(
    unit: &dfir::ProgramUnit<A>,
    comp: ComputeComp,
) -> Vec<OpId> {
    let scope: &[DfirOp] = &unit.body;

    // `unit_op.walk<WalkOrder::PreOrder>([&](Operation *op) { if (isa<…>(op)) is_visited[op] = false; })`
    // (`:1163-1166`) — ⛔ SEEDED FOR ALL THREE CLASSES BEFORE ANY IS ASKED, so entry 341 can see a
    // sibling store it has not reached yet.
    let mut is_visited: BTreeMap<OpId, bool> = BTreeMap::new();
    walk_positions(scope, &[], 0, &mut |op, at| -> Option<()> {
        if non_compute_legality(op) == NonComputeLegality::Dynamic {
            is_visited.insert(at, false);
        }
        None
    });

    let mut to_fuse: Vec<OpId> = Vec::new();
    walk_positions(scope, &[], 0, &mut |op, at| -> Option<()> {
        // `send_op.getSendData()` / `store_op.getValueToStore()` (`:1205-1211`) — ⛔ `Operation *data`
        // is left UNINITIALISED by an `else`-less chain, which only the target's three-op
        // registration makes safe.
        let data = match op {
            DfirOp::Dataflow(dataflow::Op::Send { data, .. })
            | DfirOp::Vector(vector::Op::Store { value: data, .. })
            | DfirOp::Agen(agen::Op::VectorStore { value: data, .. }) => *data,
            _ => return None,
        };

        // `VectorOperand::getOperandWithPrecision(dcc_ext_ctx, data, comp, is_precision_converted)`
        // (`:1213-1214`), whose `traverse_upwards` defaults to `true` (`VectorOperands.hpp:87-90`).
        let Some(from) = defining_position(data, scope)
            .and_then(|at| VectorOperand::operand::<A>(&at, comp, true, scope))
        else {
            return None;
        };

        // `isa<dataflow::ReceiveOp, vector::LoadOp, agen::VectorLoadOp, arith::ConstantOp>(from.op_)`
        // (`:1215-1217`) — anything else is legal and stays.
        let fusible = matches!(
            op_at(&from.op, scope),
            Some(
                DfirOp::Dataflow(dataflow::Op::Receive { .. })
                    | DfirOp::Vector(vector::Op::Load { .. })
                    | DfirOp::Agen(agen::Op::VectorLoad { .. })
            )
        ) || is_arith_constant(&from.op, scope);
        if !fusible {
            return None;
        }

        // `analyzeNonComputeOpsForFusion(unit_op, …)` (`:1222-1225`). ⭐ `unit_op` IS THE SCOPE'S OWN
        // ROOT, so entry 341's containment test is the empty prefix.
        let mut analysis = FusionAnalysis::default();
        analyze_non_compute_ops_for_fusion::<A>(
            &OpId::at(&[]),
            &at,
            &from,
            comp,
            &mut is_visited,
            scope,
            &mut analysis,
        );
        // `return !is_fusion_respected;`
        if analysis.is_fusion_respected {
            to_fuse.push(at);
        }
        None
    });

    to_fuse
}

/// Replaces: e366_fuseNonComputeOps
///
/// **366/384** `VectorChainToSentientPESFPLoweringPass::fuseNonComputeOps` —
/// `VectorChainToSentientPESFP.cpp:1159` (80L): fuse one unit's sends and stores into the compute
/// that produces the data they move.
///
/// ⛔ THE THREE PATTERNS ALL FUNNEL INTO ENTRY 364, WHICH IS NOW PORTED
/// ([`pattern_agnostic_fuse_non_compute_ops_helper`], and [`match_and_rewrite`] is `SendOpLowering`'s
/// half of the funnel) — SO WHAT IS LEFT HERE IS THIS DRIVER'S OWN WIRING, not a missing rewrite.
/// Three things it does not carry yet: `is_visited` must be ONE map spanning the legality queries and
/// the rewrites, because `applyPartialConversion` interleaves them per op while
/// [`non_compute_ops_to_fuse`] owns a map of its own and drops it; entry 364 writes through
/// `reuse_info` and mints values, so the signature needs `&mut OperandReuse` and `&mut Values`; and
/// its emission and `to_erase` are POSITIONS measured on the unrewritten body, so this driver must
/// return them for a caller to splice rather than apply them to a `&` unit. The pass above is blocked
/// before any of it — [`run_on_operation`] stops at entry 227 (`OperandReuse`), so the wiring arrives
/// with the changeset that lands it.
/// ⭐ `comp` IS THE CALLER'S. The reference reads it from `unit_op.getUnits()[0]`'s defining
/// `dataflow.get_unit` under a `DT_CHECK` (`:1173-1176`); a [`dfir::ProgramUnit`] carries it.
pub fn fuse_non_compute_ops<A: Arch>(
    unit: &dfir::ProgramUnit<A>,
    comp: ComputeComp,
    reuse_info: &OperandReuse,
) {
    let to_fuse = non_compute_ops_to_fuse::<A>(unit, comp);

    // ⭐ A UNIT WHOSE SENDS AND STORES ALL DECLINED THE FUSION IS CONVERTED SUCCESSFULLY AND
    // UNCHANGED — `applyPartialConversion` had nothing to legalize and returned `success()`.
    if let Some(first) = to_fuse.first() {
        todo!(
            "e366_fuseNonComputeOps is not wired to e364 yet — it needs one shared `is_visited`, a \
             `&mut OperandReuse` and a `&mut Values`, so {} non-compute op(s) on {:?} cannot be \
             fused (first {:?}, rest {:?}); reuse info at entry: {:?}",
            to_fuse.len(),
            comp,
            first,
            &to_fuse[1..],
            reuse_info
        );
    }
}
