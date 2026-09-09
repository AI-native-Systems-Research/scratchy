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

//! `VectorChainHelper.cpp` — 23 of bridge 2's 384 functions (dependency level(s) [0, 1, 2, 3, 6]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e059_isSentientBinaryLogicalOp` | 059/384 | 4 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:30` |
//! | `e060_getInputPrecisionFromOperand` | 060/384 | 6 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:36` |
//! | `e061_getResultPrecisionFromOperands` | 061/384 | 10 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:52` |
//! | `e062_getComputePrecisionOfOp` | 062/384 | 13 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:65` |
//! | `e063_hasConstantBounds` | 063/384 | 10 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:80` |
//! | `e064_size` | 064/384 | 6 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:319` |
//! | `e065_fuseCompareAndSelectIntoMinOrMax` | 065/384 | 49 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:415` |
//! | `e066_resetSentientFMAsIfExists` | 066/384 | 13 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:468` |
//! | `e067_redefineConstantVectors` | 067/384 | 37 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:571` |
//! | `e068_getVectorBinaryToSentientBinary` | 068/384 | 17 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:32` |
//! | `e069_getVectorElementWiseCompareOperatorToSentientBinaryOperator` | 069/384 | 9 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:55` |
//! | `e070_getVectorTernaryToSentientTernary` | 070/384 | 7 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:247` |
//! | `e163_getMaskValueConstantForNonPT` | 163/384 | 40 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:93` |
//! | `e164_checkValidityOfPackAndShuffleLowering` | 164/384 | 45 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:140` |
//! | `e165_getMergeTypeFromIndices` | 165/384 | 109 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:301` |
//! | `e228_validateLoweringAndSetMissingParameters` | 228/384 | 49 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:483` |
//! | `e229_getMaskValueForNonPT` | 229/384 | 9 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:114` |
//! | `e230_convertStringToType` | 230/384 | 24 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:196` |
//! | `e231_convertTypeToString` | 231/384 | 22 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:223` |
//! | `e277_getGCVTorFCVTTypeFromIndicesAndCastInputs` | 277/384 | 108 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:188` |
//! | `e340_analyzeAndFillOperandForwarding` | 340/384 | 25 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:535` |
//! | `e341_analyzeNonComputeOpsForFusion` | 341/384 | 86 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:610` |
//! | `e342_analyzeAndFillResultForwarding` | 342/384 | 27 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:162` |
//!
//! Original files homed here: `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp`, `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp`

use super::tf_cfgs_dataflow_conditional_tree::BlockArgEquivalence;
use super::tf_program_units_reduction::HighPreference;
use super::vc_operand_reuse::{DataId, DataOriginId, OperandReuse};
use super::vc_vector_operands::VectorOperand;
use super::vc_vector_operands::{
    ComputeComp, OpId, VectorOperandType, is_arith_constant, is_intermediate, op_at, use_positions,
};
use crate::arch::Arch;
use crate::formats::Bits;
use crate::islands::dataflow_ir::dialects::vectorchain as vc;
use crate::islands::dataflow_ir::dialects::{self as dfir_op, Op as DfirOp, Val};
use crate::islands::dataflow_ir::ty::{BoundType, ElemType, IntegerSet, ScalarTy, Vector};
use crate::islands::dataflow_ir::{Program as DfirProgram, Values};
use crate::islands::sentient::ProgramUnit as SentientProgramUnit;
use crate::islands::sentient::dialects::Op as SenOp;
use crate::islands::sentient::dialects::sentient as sen;
use std::collections::BTreeMap;

/// Replaces: e059_isSentientBinaryLogicalOp
///
/// ⭐ FOUR BITWISE OPERATORS OUT OF TWENTY — `is_any_of(sb, and0, or0, xnor, and_not)`
/// (`VectorChainHelper.cpp:30-34`). ⛔ `and0` AND `or0`, NOT `and`/`or`: the enumerators carry the
/// trailing zero (see [`sen::BinaryOp::And`]), and a reader completing the obvious spelling names
/// nothing.
///
/// ⛔⛔ TOTAL OVER THE ENUM, NO WILDCARD, AND THAT IS THE POINT OF PORTING IT AT ALL. `is_any_of`
/// over a four-element list answers *false* for every operator nobody thought about, so a
/// twenty-first `SentientBinaryOperator` that IS bitwise reads as arithmetic and the compute gets the
/// wrong precision treatment silently. Written out, a new variant stops the build here and has to
/// decide.
#[must_use]
pub const fn is_sentient_binary_logical_op(sb: sen::BinaryOp) -> bool {
    match sb {
        sen::BinaryOp::And | sen::BinaryOp::Or | sen::BinaryOp::Xnor | sen::BinaryOp::AndNot => {
            true
        }
        sen::BinaryOp::Min
        | sen::BinaryOp::Max
        | sen::BinaryOp::AbsMin
        | sen::BinaryOp::AbsMax
        | sen::BinaryOp::Add
        | sen::BinaryOp::Mul
        | sen::BinaryOp::Sub
        | sen::BinaryOp::MulDiv2
        | sen::BinaryOp::GcvtImm(_)
        | sen::BinaryOp::FcvtImm(_)
        | sen::BinaryOp::Merge { .. }
        | sen::BinaryOp::Pack(_)
        | sen::BinaryOp::CompareNeq
        | sen::BinaryOp::CompareEq
        | sen::BinaryOp::CompareLt
        | sen::BinaryOp::CompareLe => false,
    }
}

/// Replaces: e060_getInputPrecisionFromOperand
///
/// ⭐ IT READS ONE FIELD — `precision = operand.orig_precision_` (`VectorChainHelper.cpp:36-42`) —
/// AND THE ONE THING IT DOES ON THE WAY OUT IS ALREADY DONE BY THE TYPE.
///
/// ⛔⛔ THE `fp80` REMAP IS NOT DROPPED, IT IS UNREPRESENTABLE, WHICH IS STRONGER. The reference
/// carries `if (precision == "fp80") precision = "fp8";` with the comment *"Currently, we use fp80
/// type in MLIR to represent fp8"* — MLIR's `Float80Type` is a stand-in for fp8 because MLIR has no
/// 8-bit float of the right flavour, and `VectorOperands.cpp:227-232` does the same trick on the
/// width (`if (bit_width == 80) bit_width = 8; else if (bit_width == 24) bit_width = 16;`).
/// [`sen::Precision`] has no `Fp80` case and [`crate::islands::dataflow_ir::ty::ElemType`] has no
/// F80, so the stand-in never enters this crate and the remap has nothing left to fire on. ⛔ Adding
/// an `Fp80` case to reproduce the string would reintroduce the very hazard the remap exists to
/// paper over.
///
/// ⛔ `None` IS THE ABSENT PRECISION AND [`sen::Precision::None`] IS NOT IT. The C++'s
/// `std::string` answers `""` for an operand nothing set a precision on, and `""` is what
/// `validateLoweringAndSetMissingParameters` tests to decide whether to substitute the compute
/// precision (`VectorChainToSentientPESFP.cpp:257-263`). `Precision::None` is a precision that
/// prints `none`, a different thing entirely — so the emptiness lives in the [`Option`].
///
/// ⭐ THE `std::optional` OVERLOAD AT `:44-50` IS THIS FUNCTION ON AN `Option`, and it is not a
/// scheduled unit. `operand.as_ref().and_then(input_precision_from_operand)` is it, so nothing is
/// missing.
#[must_use]
pub fn input_precision_from_operand(operand: &VectorOperand) -> Option<sen::Precision> {
    operand.orig_precision
}

/// Replaces: e061_getResultPrecisionFromOperands
///
/// ⛔⛔ IT RETURNS ON THE FIRST **PRESENT** OPERAND, NOT ON THE FIRST ONE THAT HAS A PRECISION, AND
/// THAT IS NOT WHAT THE OBVIOUS PORT WRITES. The loop body is inside
/// `if (opr_.has_value()) { … return precision; }` (`VectorChainHelper.cpp:52-62`) — the `return` is
/// unconditional once an operand exists — so a first operand carrying an EMPTY
/// `on_the_fly_conv_precision_` makes the whole function answer empty and the later operands are
/// never consulted. `.filter_map(|o| o.on_the_fly_conv_precision).next()` would skip past it and
/// answer with a later operand's precision, which is a different function.
///
/// ⛔ AND THE TRAILING `return "";` IS MISSING FROM THE EXTRACT (`bridge2.cpp` truncates 366 of 384
/// bodies at the tail). An all-`nullopt` list answers empty, not the first operand's anything.
///
/// ⭐ `on_the_fly_conv_precision_`, NOT `orig_precision_` — the sibling
/// [`input_precision_from_operand`] reads the other field. The result of a compute is at the
/// precision its operands were CONVERTED to on the way in, not the one they were stored at; reading
/// `orig_precision_` here would emit the input precision as the output's and typecheck perfectly.
///
/// The `fp80` remap collapses into the type exactly as it does for the input precision; see
/// [`input_precision_from_operand`].
#[must_use]
pub fn result_precision_from_operands(
    operands: &[Option<VectorOperand>],
) -> Option<sen::Precision> {
    operands
        .iter()
        .flatten()
        .next()
        .and_then(|operand| operand.on_the_fly_conv_precision)
}

/// Replaces: e062_getComputePrecisionOfOp
///
/// ⛔⛔ THE `pack`/`merge` SHORT-CIRCUIT COMES FIRST AND IS NOT AN OPTIMISATION.
/// `if (isa<PackOp, MergeOp>(op)) return "fp16";` (`VectorChainHelper.cpp:65-68`) runs BEFORE
/// `getElementType`, and for a [`vc::Op::Merge`] that ordering is the only thing standing between
/// this function and an abort: `MergeOp` appears in NEITHER `getVectorType` NOR
/// `getCustomVectorType` (`Utils.cpp:538-694`), so asking one for its element type hits
/// `DT_CHECK_MSG(.., "Type is not a known vector type")`. Reordering the branches — or "simplifying"
/// them away because a pack's element type looks like it would do — turns a working lowering into a
/// crash on merges and a wrong answer on packs.
///
/// ⭐ AND IT IS A WRONG ANSWER, DEMONSTRABLY. IBM's own `gcvt.mlir` packs two
/// `vector<128xf8E4M3FN>` operands and the lowered compute carries
/// `ComputePrecision = #sentient<precision fp16>`
/// (`dcc/test/Conversion/VectorChainToSentientPESFP/gcvt.mlir`) — `fp16` where the element type says
/// `fp8`, because a pack computes in the wide format and narrows on the way out.
///
/// ⚠️ THE CALLER MAY OVERRIDE IT BACK TO `none`, and that is the caller's decision, not this
/// function's. `PackOpLowering` sets `op_info.compute_precision_ = "none"` when NEITHER operand is a
/// [`vc::Op::Cast`] — *"MERGE/PACK instructions are bitwise operations so compute precisions should
/// be set to none"* (`VectorChainToSentientPESFP.cpp:702-705`) — which is why the merge/pack computes
/// in `dcc/test/SFP/merge_and_pack.mlir` read `ComputePrecision = none` while the gcvt ones read
/// `fp16`. Answering `none` here would break the convert case.
///
/// ⭐ `bf16` REMAPS TO `fp16`; `dlfp16` CANNOT ARRIVE. The reference remaps both
/// (`is_any_of(precision, "bf16", "dlfp16")`), but `getPrecisionInString` can only ever produce
/// `int<n>`, `bf16`, `mxfp<n>` or `fp<n>` — never the string `dlfp16` — so that half of the
/// condition is dead in the reference too. See [`precision_in_string`].
///
/// The `fp80` arm collapses into the type exactly as it does for the operand precisions; see
/// [`input_precision_from_operand`].
#[must_use]
pub fn compute_precision_of_op(op: &DfirOp) -> sen::Precision {
    // ⛔ BEFORE THE ELEMENT-TYPE QUERY. See the note above.
    if matches!(
        op,
        DfirOp::VectorChain(vc::Op::Pack { .. } | vc::Op::Merge { .. })
    ) {
        return sen::Precision::Fp16;
    }

    // ⛔ TOTAL OVER THE ENUM, NO WILDCARD, so a precision added later has to say whether the
    // reference's remap list grew with it. Only `Bf16` moves; every other case is itself.
    match precision_in_string(element_type_of(op)) {
        sen::Precision::Bf16 => sen::Precision::Fp16,
        precision @ (sen::Precision::Int1
        | sen::Precision::Int2
        | sen::Precision::Int4
        | sen::Precision::Int8
        | sen::Precision::Int16
        | sen::Precision::Int24
        | sen::Precision::Int32
        | sen::Precision::Int64
        | sen::Precision::Mxfp4
        | sen::Precision::Mxfp8
        | sen::Precision::Mxint4
        | sen::Precision::Fp4
        | sen::Precision::Fp8
        | sen::Precision::Fp16
        | sen::Precision::IeeeFp16
        | sen::Precision::Fp24
        | sen::Precision::Fp32
        | sen::Precision::None) => precision,
    }
}

/// WHAT ONE ELEMENT TYPE IS CALLED AS A PRECISION — `dataflow::utils::getPrecisionInString`
/// (`dialect_utils/Dataflow/Utils.cpp:187-203`).
///
/// ⭐ NOT A SCHEDULED UNIT: it is one of the campaign's 106 exclusions, supplied here because
/// [`compute_precision_of_op`] cannot answer without it. It builds a STRING in the reference —
/// `"int"`, `"bf"` or `"fp"` concatenated with the bit width — which the caller then hands to
/// `symbolizeSentientPrecision`; the two steps collapse into one [`sen::Precision`] here, and the
/// crate's own rule is why (no strings for closed sets).
///
/// ⛔⛔ `f16` IS `fp16`, NOT `ieee_fp16`, however much the MLIR type looks like the IEEE one. The
/// reference's whole float branch is `"fp" + getIntOrFloatBitWidth()`, so `ieee_fp16` — a real
/// [`sen::Precision`] case — is UNREACHABLE from here. Answering it would name the 1-6-9 format the
/// hardware does not compute in.
///
/// ⭐ AND `bf16` IS RETURNED AS ITSELF, not pre-remapped. `getPrecisionInString` has no idea what
/// its caller will do with it; [`compute_precision_of_op`] is the one that turns it into `fp16`, and
/// other callers of the reference's function do not.
pub(super) fn precision_in_string(elem: ElemType) -> sen::Precision {
    match elem {
        // ⭐ THE MX CHECK IS FIRST IN THE REFERENCE, before `isIntOrIndexOrFloat` is even asserted:
        // a `CustomMXFloatType` is not an int-or-float, so the order is load-bearing there. Here the
        // element type already distinguishes them.
        ElemType::MxFloat(4) => sen::Precision::Mxfp4,
        ElemType::MxFloat(8) => sen::Precision::Mxfp8,
        ElemType::MxInt(4) => sen::Precision::Mxint4,
        ElemType::Int(1) => sen::Precision::Int1,
        ElemType::Int(2) => sen::Precision::Int2,
        ElemType::Int(4) => sen::Precision::Int4,
        ElemType::Int(8) => sen::Precision::Int8,
        ElemType::Int(16) => sen::Precision::Int16,
        // ⛔ `int24` NAMES A SIXTEEN-BIT REGISTER, and that is `SentientTypes.td`'s business, not a
        // reason to fold it into `Int16`: `getPrecisionInString` writes `"int" + 24` for an `i24`
        // and the PT's accumulator format IS `i24` (`SNComputeLowering.cpp:291-297`).
        ElemType::Int(24) => sen::Precision::Int24,
        ElemType::Int(32) => sen::Precision::Int32,
        ElemType::Int(64) => sen::Precision::Int64,
        ElemType::F16 => sen::Precision::Fp16,
        ElemType::Bf16 => sen::Precision::Bf16,
        ElemType::F32 => sen::Precision::Fp32,
        // ⭐ BOTH FP8 FLAVOURS ARE `fp8`. The width is all the reference reads, so `f8E4M3FN` and
        // `f8E8M0FNU` are one precision — which is also why `SEN053_FP8` can borrow E4M3's type.
        ElemType::F8E4M3Fn | ElemType::F8E8M0Fnu | ElemType::F8E5M2 => sen::Precision::Fp8,
        ElemType::F4E2M1Fn => sen::Precision::Fp4,
        // ⛔ NO SUCH PRECISION EXISTS TO NAME. `SentientPrecisionAttr` has nineteen cases and none of
        // them is an `int<n>` or `mxfp<n>` for any other `n`; the reference would build the string
        // and then abort in `symbolizeSentientPrecision(..).value()`
        // (`VectorChainToSentientPESFP.cpp:362`).
        ElemType::Int(bits) | ElemType::MxFloat(bits) | ElemType::MxInt(bits) => {
            todo!("no sentient precision names a {bits}-bit element")
        }
    }
}

/// THE ELEMENT TYPE AN OP'S VECTOR IS MADE OF — `dcc::utils::getElementType`
/// (`Utils.cpp:695-702`).
///
/// ⭐ NOT A SCHEDULED UNIT, and one function here where the reference has three:
/// `getElementType` tries `getVectorType` and falls back to `getCustomVectorType`, whose only
/// difference is whether the element is an MX format — and [`ElemType::MxFloat`] already carries
/// that, so [`vector_type_of`] answers for both.
///
/// ⛔ IT ABORTS ON AN OP OUTSIDE ITS DOMAIN, and so does this. `DT_CHECK_MSG(custom_vtype.has_value(),
/// "Type is not a known vector type")` is unconditional; there is no precision to answer with, and
/// inventing one would be the stand-in the crate's rules forbid.
fn element_type_of(op: &DfirOp) -> ElemType {
    match vector_type_of(op) {
        Some(ty) => ty.elem,
        None => todo!("getElementType: {op:?} has no vector type to take an element from"),
    }
}

/// THE VECTOR TYPE AN OP CARRIES, IF IT IS ONE OF THE OPS THAT CARRIES ONE — `getVectorType` and
/// `getCustomVectorType` (`Utils.cpp:538-694`) as one function.
///
/// ⛔⛔ THE MEMBERSHIP LIST IS THE WHOLE CONTENT, AND IT IS NOT "EVERY OP WITH A VECTOR RESULT".
/// [`vc::Op::Cast`], [`vc::Op::Select`], [`vc::Op::Rotate`], [`vc::Op::ConstantBitstream`],
/// [`vc::Op::CreateAffineMask`], [`vc::Op::CreateAffineMaskSet`] and [`vc::Op::Merge`] every one of
/// them results in a vector, and
/// every one of them is ABSENT from both reference lists — so `getElementType` aborts on a cast.
/// ⛔ `RotateOp` IS THE ONE TO WATCH: `getLoadConsumer` names it beside `SelectOp` and `ShuffleOp`
/// as a rearrangement (`Helper.cpp:1268-1270`), and of those three only the SHUFFLE is in
/// `getVectorType` (`Utils.cpp:581`). `RotateOp` appears nowhere in `Utils.cpp` at all. Answering from the result type
/// "because it obviously has one" would paper over the abort the reference relies on
/// [`compute_precision_of_op`] short-circuiting past.
///
/// ⭐ `vector::LoadOp` AND `vector::StoreOp` ARE ON THE LIST (`Utils.cpp:548-553`, between the
/// `dataflow` transfer arms and the `agen` access arms), and they now exist here as
/// [`dfir_op::vector::Op`] — so those two answer from their own `ty` directly.
///
/// ⚠️ ONE DELIBERATE WIDENING, FLAGGED, AND IT IS THE AFFINE PAIR: this island also holds the same
/// two accesses in their pre-D1 `affine` form ([`dfir_op::affine::Op::VectorLoad`]/`VectorStore`),
/// which the reference has no op class for because `AffineToStandard` has already rewritten them by
/// the time `getVectorType` runs. They are the same access one rung earlier; treating them as absent
/// would make the answer depend on which side of D1 the query happens to run.
pub(super) fn vector_type_of(op: &DfirOp) -> Option<Vector> {
    match op {
        // `dataflow::SendOp` and `dataflow::ReceiveOp`.
        DfirOp::Dataflow(dfir_op::dataflow::Op::Send { ty, .. }) => Some(*ty),
        DfirOp::Dataflow(dfir_op::dataflow::Op::Receive { ty, .. }) => Some(*ty),
        DfirOp::Dataflow(
            dfir_op::dataflow::Op::GetUnit { .. }
            | dfir_op::dataflow::Op::GetLocalUnit { .. }
            | dfir_op::dataflow::Op::CreateGroup { .. }
            | dfir_op::dataflow::Op::GetLogicalMemoryView { .. }
            | dfir_op::dataflow::Op::GetPagedLogicalMemoryView(_)
            | dfir_op::dataflow::Op::ProgramUnit { .. }
            // The collection binders bind an `index` and a `vector<Nxindex>`, never a vector of
            // data — see [`dfir_op::dataflow::Op::GetUnitCollection`].
            | dfir_op::dataflow::Op::GetUnitCollection { .. }
            | dfir_op::dataflow::Op::GetMyUnitInCollection { .. }
            | dfir_op::dataflow::Op::GetTotalUnitsInCollection { .. }
            | dfir_op::dataflow::Op::ProgramCollection { .. }
            | dfir_op::dataflow::Op::SyncSend { .. }
            | dfir_op::dataflow::Op::SyncRecv { .. }
            | dfir_op::dataflow::Op::ImplicitSync { .. }
            | dfir_op::dataflow::Op::Opaque(_),
        ) => None,
        // `agen::VectorLoadOp` and `agen::VectorStoreOp`.
        DfirOp::Agen(dfir_op::agen::Op::VectorLoad { ty, .. }) => Some(*ty),
        DfirOp::Agen(dfir_op::agen::Op::VectorStore { ty, .. }) => Some(*ty),
        // ⛔ AND THE SYMBOLIC PAIR IS NOT ON THE LIST. `getVectorType`'s `dyn_cast` chain names
        // `agen::VectorLoadOp` and `agen::VectorStoreOp` only (`Utils.cpp:551-556`); no
        // `SymbolicVector*Op` appears in `Utils.cpp` at all, so it aborts on one exactly as it does
        // on a `RotateOp`.
        DfirOp::Agen(
            dfir_op::agen::Op::SymbolicVectorLoad { .. }
            | dfir_op::agen::Op::SymbolicVectorStore { .. }
            | dfir_op::agen::Op::IndirectVectorLoad { .. }
            | dfir_op::agen::Op::IndirectVectorStore { .. }
            | dfir_op::agen::Op::CompositeLoad(_)
            | dfir_op::agen::Op::CompositeStore(_)
            | dfir_op::agen::Op::CompositeIndirectLoad(_)
            | dfir_op::agen::Op::CompositeIndirectStore(_)
            | dfir_op::agen::Op::CompositeLoadAndStore(_)
            | dfir_op::agen::Op::CompositeIndirectLoadAndStore(_)
            // ⛔ AND THE MASK STATE IS NOT ON IT EITHER, THOUGH IT DOES BIND A VECTOR: the chain
            // names the two accesses only, so a mask state aborts exactly as a symbolic one does.
            | dfir_op::agen::Op::CompositeMemoryInterleave { .. }
            | dfir_op::agen::Op::SetTransferMaskState { .. }
            | dfir_op::agen::Op::Yield,
        ) => None,
        // `vector::LoadOp` and `vector::StoreOp` (`Utils.cpp:548-553`).
        DfirOp::Vector(dfir_op::vector::Op::Load { ty, .. }) => Some(*ty),
        DfirOp::Vector(dfir_op::vector::Op::Store { ty, .. }) => Some(*ty),
        // The same two accesses in their pre-D1 `affine` form — see the widening note.
        DfirOp::Affine(dfir_op::affine::Op::VectorLoad { ty, .. }) => Some(*ty),
        DfirOp::Affine(dfir_op::affine::Op::VectorStore { ty, .. }) => Some(*ty),
        DfirOp::Affine(
            dfir_op::affine::Op::For { .. }
            | dfir_op::affine::Op::Apply { .. }
            | dfir_op::affine::Op::Yield { .. }
            // ⭐ AN `affine.if` YIELDS `index`, NEVER A VECTOR — `%52 = affine.if #set0(%arg9) -> index`
            // (`dcc/test/PT/issue-236.mlir:59`). It is absent from both reference lists for the same
            // reason it answers `None` here.
            | dfir_op::affine::Op::If { .. },
        ) => None,
        // The `vectorchain` ops both reference lists name, including the estimate family:
        // `ExpEstimateOp` (`Utils.cpp:598`), `RecEstimateOp`, `LnEstimateOp`, `RsqrtEstimateOp`,
        // `SigmoidEstimateOp` and `TanhEstimateOp`, which [`vc::Op::Estimate`] holds as one op, plus
        // `FastExpOp` (`:594`) and `FloorOp` (`:615`), which are their own ops in that dialect and
        // now their own variants here — see [`vc::Op::FastExp`].
        DfirOp::VectorChain(
            vc::Op::Estimate { ty, .. }
            | vc::Op::FastExp { ty, .. }
            | vc::Op::Floor { ty, .. }
            | vc::Op::ScanWithGap { ty, .. }
            | vc::Op::Multiply { ty, .. }
            | vc::Op::MultiplyAccumulate { ty, .. }
            | vc::Op::ElementWiseCompare { ty, .. }
            | vc::Op::ElementWiseSelection { ty, .. }
            | vc::Op::Binary { ty, .. }
            | vc::Op::Shuffle { ty, .. }
            | vc::Op::Pack { ty, .. },
        ) => Some(*ty),
        // ⛔ THE SIX THAT RESULT IN A VECTOR AND ARE STILL ABSENT. See the note above.
        DfirOp::VectorChain(
            vc::Op::Select { .. }
            // ⛔ `NegOp` IS ABSENT FROM BOTH REFERENCE LISTS TOO — it appears nowhere in
            // `dcc/src/Utils/Utils.cpp` at all, exactly like `RotateOp`, even though
            // `(outs AnyVectorOfAnyRank:$data)` gives it a vector result.
            | vc::Op::Neg { .. }
            | vc::Op::Rotate { .. }
            | vc::Op::ConstantBitstream { .. }
            | vc::Op::Cast { .. }
            | vc::Op::CreateAffineMask { .. }
            | vc::Op::CreateAffineMaskSet { .. }
            | vc::Op::Merge { .. },
        ) => None,
        // Neither reference list mentions an `arith`, an `scf` or a `symbol` op.
        // Neither reference list mentions an `arith`, an `scf`, a `symbol` or a `uniform` op.
        // ⭐ `uniform.uniformize_regions` MAY BIND RESULTS and none of them is a vector: its own
        // verifier restricts them to `index` (`Uniform.cpp:99-121` prints exactly that type).
        DfirOp::Arith(_) | DfirOp::Scf(_) | DfirOp::Symbol(_) | DfirOp::Uniform(_) => None,
    }
}

/// Replaces: e063_hasConstantBounds
///
/// WHETHER A MASK SET PINS ITS ONE DIMENSION BETWEEN TWO CONSTANTS — the gate
/// `getMaskValueConstantForNonPT` opens before it divides by anything
/// (`VectorChainHelper.cpp:80-90`, its sole caller at `:112`).
///
/// ⛔ POSITION ZERO IS THE C++'S OWN LITERAL, and it is only meaningful because the caller has
/// already refused any set with more than one dimension or with symbols (`:100-107`). This asks
/// about the mask's single lane axis.
///
/// ⛔⛔ BOTH SIDES, IN THAT ORDER, AND NOT `Lb <= Ub`. A one-sided set answers `false` on the second
/// check; a set whose bounds CROSS answers `true`, because both constants are there. IBM depends on
/// the crossing case: `affine_set<(d0) : (d0 - 64 >= 0, -d0 + 63 >= 0)>` is the all-lanes-off mask of
/// twenty ops in `dcc/test/Conversion/VectorChainToSentientPESFP/mixed_precision.mlir`, and it
/// lowers, so tightening this to a non-emptiness test would reject working input. See
/// [`IntegerSet::constant_bound`].
///
/// ⛔ THE `return true;` TAIL IS MISSING FROM `bridge2.cpp`'S EXTRACT — the authority has it at
/// `:90`. Without it the function would read as "false unless proven", which is the same answer only
/// by accident.
#[must_use]
pub fn has_constant_bounds(mask_set_flat: &IntegerSet) -> bool {
    if mask_set_flat.constant_bound(BoundType::Lb, 0).is_none() {
        return false;
    }

    if mask_set_flat.constant_bound(BoundType::Ub, 0).is_none() {
        return false;
    }

    true
}

/// WHICH OF THE THIRTY-FOUR MERGE-OR-PACK INSTRUCTIONS a table row names.
///
/// ⛔ NOT A STRING, AND NOT [`sen::BinaryOp`] EITHER. The C++ row carries
/// `const std::string name` and `getMergeTypeFromIndices` returns it, so a caller can compare it
/// against anything at all; and the twenty-operator `BinaryOp` would let `and0` be written into a
/// pack table. Exactly two shapes reach that table — `merge<w><half>` and `pack<n>` — so exactly two
/// are spellable here, and [`sen::PackIndex`] already refuses the `pack10`/`pack11` that do not
/// exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeOrPack {
    /// `merge8h`, `merge8l`, `merge16h`, … — eight rows of the table.
    Merge {
        /// The element width the merge interleaves at.
        width: sen::MergeWidth,
        /// Whether it is the high half rather than the low.
        high: bool,
    },
    /// `pack0` … `pack27` — twenty-six rows of the table.
    Pack(sen::PackIndex),
}

impl MergeOrPack {
    /// THE SENTIENT BINARY OPERATOR THIS NAMES.
    ///
    /// ⭐ THE MNEMONIC IS ALREADY [`sen::BinaryOp::spelling`]'s JOB, so the row's `name_` string is
    /// this plus that call — never a second spelling of `merge{bits}{h|l}`.
    #[must_use]
    pub const fn as_binary(self) -> sen::BinaryOp {
        match self {
            MergeOrPack::Merge { width, high } => sen::BinaryOp::Merge { width, high },
            MergeOrPack::Pack(index) => sen::BinaryOp::Pack(index),
        }
    }
}

/// ONE ROW OF THE MERGE-AND-PACK TABLE — `merge_and_pack_type`, the local struct
/// `getMergeTypeFromIndices` matches a pack's indices against
/// (`VectorChainHelper.cpp:304-325`).
///
/// ⛔ ITS FIELDS ARE PRIVATE BECAUSE TWO OF THEM ARE DERIVED. `size` and `sum` are computed from
/// `vec` once, in the constructor, and the C++ leaves all three public and mutable — so a row whose
/// `vec` was edited afterwards would match on a stale length. Here the only way to make a row is
/// [`MergeAndPack::new`], and the three cannot disagree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeAndPack {
    /// Which instruction this row names.
    name: MergeOrPack,
    /// The element width the row's indices are written for.
    element_bit_width: Bits,
    /// The lane indices, where `-1` is a lane the instruction does not write.
    vec: Vec<i32>,
    /// Whether the instruction sign-extends.
    sign_extend: bool,
    /// How many times the pattern repeats.
    repetition: u32,
    /// `vec.size()`.
    size: usize,
    /// The sum of `vec`.
    sum: i32,
}

impl MergeAndPack {
    /// Replaces: e064_size
    ///
    /// A ROW, WITH ITS LENGTH AND ITS CHECKSUM DERIVED FROM ITS INDICES — the
    /// `merge_and_pack_type` constructor (`VectorChainHelper.cpp:312-324`).
    ///
    /// ⭐ THE TWO DERIVED MEMBERS ARE WHAT THE CONSTRUCTOR IS FOR. `size(vec.size())` is a member
    /// initialiser and `sum` is an immediately-invoked lambda over the same vector
    /// (`:318`, `:319-323`); `getMergeTypeFromIndices` then rejects a candidate row on
    /// `indices.size() != inst.size * scale` and, when `scale == 1`, on
    /// `indices_sum != inst.sum` — a cheap pair of filters ahead of the per-lane comparison
    /// (`:398-400`).
    ///
    /// ⛔ `sum` IS SIGNED AND THE `-1`s COUNT. `pack24`'s vector is sixteen real lanes followed by
    /// forty-eight `-1`s, so its sum is 960 - 48 = 912, not 960. Summing only the non-negative
    /// entries would make the fast filter accept index lists it should reject.
    ///
    /// ⛔ `repetition` IS ALWAYS EIGHT. It is a member DEFAULT (`int repetition = 8;`, `:310`) that
    /// the constructor never assigns and no row overrides, and it is compared for equality against
    /// the pack op's own `repetition` (`:399`) — so it is a constant of the table, not a parameter.
    ///
    /// ⛔ `sign_extend` IS EXPLICIT AT EVERY CALL. The C++ defaults it (`bool sign_extend = false`,
    /// `:314`) and thirty-two of the thirty-four rows leave it out; Rust has no default arguments, so
    /// the two rows that DO set it — `pack14` (`:350-353`) and `pack15` (`:354-357`) — read beside
    /// `pack12` and `pack13`, the twins they differ from in nothing else, rather than differing in an
    /// omission.
    ///
    /// ⭐ ITS THIRTY-FOUR CALLERS ARE [`merge_and_pack_insts`], the table `getMergeTypeFromIndices`
    /// (entry 165) scans — which is where the two filters above are spent.
    #[must_use]
    pub fn new(
        name: MergeOrPack,
        element_bit_width: Bits,
        vec: Vec<i32>,
        sign_extend: bool,
    ) -> MergeAndPack {
        MergeAndPack {
            name,
            element_bit_width,
            size: vec.len(),
            sum: vec.iter().sum(),
            vec,
            sign_extend,
            repetition: 8,
        }
    }

    /// Which instruction this row names.
    #[must_use]
    pub const fn name(&self) -> MergeOrPack {
        self.name
    }

    /// The element width the row's indices are written for.
    #[must_use]
    pub const fn element_bit_width(&self) -> Bits {
        self.element_bit_width
    }

    /// The lane indices, where `-1` is a lane the instruction does not write.
    #[must_use]
    pub fn vec(&self) -> &[i32] {
        &self.vec
    }

    /// Whether the instruction sign-extends.
    #[must_use]
    pub const fn sign_extend(&self) -> bool {
        self.sign_extend
    }

    /// How many times the pattern repeats — eight, for every row of the table.
    #[must_use]
    pub const fn repetition(&self) -> u32 {
        self.repetition
    }

    /// How many indices the row has.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.size
    }

    /// The sum of the row's indices, `-1`s included.
    #[must_use]
    pub const fn sum(&self) -> i32 {
        self.sum
    }
}

/// WHAT [`fuse_compare_and_select_into_min_or_max`] DECIDED — the two out-bools as one answer.
///
/// ⭐⭐ THREE STATES, NOT TWO BOOLS. The reference hands back `bool& fusion_to_min` and
/// `bool& fusion_to_max` and then has to ask itself `if (!fusion_to_min && !fusion_to_max)`
/// (`VectorChainHelper.cpp:462-464`), because the pair can spell a fourth thing — *both* — that
/// means nothing at all. One value cannot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[must_use]
pub enum MinOrMaxFusion {
    /// Neither bool set: this selection is not a min or a max in disguise, and stays a ternary.
    NotFused,
    /// `fusion_to_min` — the pair lowers to one `#sentient<binary_operator min>`.
    ToMin,
    /// `fusion_to_max` — the pair lowers to one `#sentient<binary_operator max>`.
    ToMax,
}

/// Replaces: e065_fuseCompareAndSelectIntoMinOrMax
///
/// **065/384** `vectorchain::fuseCompareAndSelectIntoMinOrMax` —
/// `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:415` (49L).
///
/// ```cpp
/// void vectorchain::fuseCompareAndSelectIntoMinOrMax(
///     ElementWiseSelectionOp selection_op, bool& fusion_to_min,
///     bool& fusion_to_max) {
///   fusion_to_min = false;
///   fusion_to_max = false;
///   if (auto compare_op = llvm::dyn_cast<ElementWiseCompareOp>(
///           selection_op.getCond().getDefiningOp())) {
///     bool same_pair_matching = false, opposite_pair_matching = false;
///
///     // Operation equivalence is used since some times constant operands
///     // are duplicated, and direct match may result in spurious mismatches.
///     dcc::OperationEquivalence oe;
///
///     auto lhs_def = selection_op.getLhs().getDefiningOp();
///     auto rhs_def = selection_op.getRhs().getDefiningOp();
///     auto op1_def = compare_op.getOp1().getDefiningOp();
///     auto op2_def = compare_op.getOp2().getDefiningOp();
///     if (!lhs_def || !rhs_def || !op1_def || !op2_def) return;
///
///     if (oe.operationsAreEquivalent(*lhs_def, *op1_def, nullptr) &&
///         oe.operationsAreEquivalent(*rhs_def, *op2_def, nullptr)) {
///       same_pair_matching = true;
///     } else if (oe.operationsAreEquivalent(*lhs_def, *op2_def, nullptr) &&
///                oe.operationsAreEquivalent(*rhs_def, *op1_def, nullptr)) {
///       opposite_pair_matching = true;
///     }
///
///     if (!same_pair_matching && !opposite_pair_matching) return;
///
///     if (is_any_of(compare_op.getCompareOp(), compare_gt, compare_ge)) {
///       if (same_pair_matching)          fusion_to_max = true;
///       else if (opposite_pair_matching) fusion_to_min = true;
///     } else if (is_any_of(compare_op.getCompareOp(), compare_lt, compare_le)) {
///       if (same_pair_matching)          fusion_to_min = true;
///       else if (opposite_pair_matching) fusion_to_max = true;
///     }
///
///     if (!fusion_to_min && !fusion_to_max) {
///       DT_ERROR("Unable to lower compare and select into max/min operation");
///     }
///   }
/// }
/// ```
///
/// ⭐⭐ `max(a, b)` IS WRITTEN IN THIS IR AS `a > b ? a : b`, AND THIS IS WHERE IT IS RECOGNISED.
/// A `vectorchain.element_wise_compare` feeding a `vectorchain.element_wise_selection` whose two arms
/// are the compare's own two operands is one Sentient instruction, not two: `fmax.mlir:119-120` is
/// exactly that pair and `fmax.mlir:42` lowers it to a single
/// `sentient.vector_binary … binaryOp = #sentient<binary_operator max>`. Get this wrong and a
/// two-instruction chain is emitted where the ISA has one — or worse, `min` where the program said
/// `max`.
///
/// ⛔⛔ THE ARMS ARE COMPARED BY **STRUCTURE**, NOT BY SSA NAME, AND THAT IS LOAD-BEARING. The
/// reference's own comment says why: *"since some times constant operands are duplicated, and direct
/// match may result in spurious mismatches"*. `relu.mlir:50-51` has two separate
/// `arith.constant dense<0.000000e+00> : vector<64xf16>` ops, `%cst` and `%cst_1`; the compare reads
/// `%cst` and the selection reads `%cst_1` (`:73-74`), and it still has to fuse — `relu.mlir:35,37`
/// shows the `max`. A `lhs == op1 && rhs == op2` test on value identity answers *false* there and
/// emits the two-instruction form. See [`ops_are_equivalent`].
///
/// ⛔ THE ARGUMENT IS THE SELECTION'S THREE FIELDS, NOT A `&vc::Op`. The reference's parameter type is
/// `ElementWiseSelectionOp` — the narrowing has already happened at the call site, and taking a whole
/// [`vc::Op`] here would force a `todo!` arm for the seventeen variants the C++ signature already rules
/// out. `scope` is the block to resolve definitions in; the reference gets that from the operand's own
/// `getDefiningOp()`, which this island reaches through [`dfir_op::defining_op`].
///
/// ⭐ `NotFused` COVERS BOTH OF THE REFERENCE'S SILENT RETURNS: the cond was not a compare at all
/// (`fcmp_select.mlir:156` — the selection's cond `%62` is an `agen.vector_load`, so the pair does not
/// fuse and `:64` emits the `vector_ternary`), and an arm was a region argument with no defining op.
///
/// ⛔ ONE `todo!` FOR THE `DT_ERROR`, AND ITS DOMAIN IS EXACTLY `compare_eq`/`compare_neq`. Reaching
/// the error needs a matching pair *and* an operator outside the four ordering comparisons, which
/// leaves only equality — an `a == b ? a : b` that is neither a min nor a max. Every other route to
/// `!fusion_to_min && !fusion_to_max` already returned above.
pub fn fuse_compare_and_select_into_min_or_max(
    cond: vc::Predicate,
    lhs: Val,
    rhs: Val,
    scope: &[DfirOp],
) -> MinOrMaxFusion {
    // `llvm::dyn_cast<ElementWiseCompareOp>(selection_op.getCond().getDefiningOp())` — a cond
    // defined by anything else, or by nothing, leaves both bools false.
    let Some(DfirOp::VectorChain(vc::Op::ElementWiseCompare {
        op1,
        op2,
        compare_op,
        ..
    })) = dfir_op::defining_op(cond.val(), scope)
    else {
        return MinOrMaxFusion::NotFused;
    };

    // `if (!lhs_def || !rhs_def || !op1_def || !op2_def) return;`
    let (Some(lhs_def), Some(rhs_def), Some(op1_def), Some(op2_def)) = (
        dfir_op::defining_op(lhs, scope),
        dfir_op::defining_op(rhs, scope),
        dfir_op::defining_op(*op1, scope),
        dfir_op::defining_op(*op2, scope),
    ) else {
        return MinOrMaxFusion::NotFused;
    };

    // `dcc::OperationEquivalence oe;` — the three-argument constructor, which leaves `functor_`
    // default-constructed and so null.
    let no_functor = HighPreference::None;
    // `all_block_args_are_equiv` is the same constructor's third default.
    let any_block_arg = BlockArgEquivalence::AllEquivalent;
    let same_pair_matching = ops_are_equivalent(lhs_def, op1_def, scope, no_functor, any_block_arg)
        && ops_are_equivalent(rhs_def, op2_def, scope, no_functor, any_block_arg);
    let opposite_pair_matching = !same_pair_matching
        && ops_are_equivalent(lhs_def, op2_def, scope, no_functor, any_block_arg)
        && ops_are_equivalent(rhs_def, op1_def, scope, no_functor, any_block_arg);

    // `if (!same_pair_matching && !opposite_pair_matching) return;`
    if !same_pair_matching && !opposite_pair_matching {
        return MinOrMaxFusion::NotFused;
    }

    // Past that guard `!same_pair_matching` IS `opposite_pair_matching`, so one bool decides which
    // way round the arms sit and the reference's second `else if` needs no counterpart.
    match (compare_op, same_pair_matching) {
        (vc::CompareOp::Gt | vc::CompareOp::Ge, true) => MinOrMaxFusion::ToMax,
        (vc::CompareOp::Gt | vc::CompareOp::Ge, false) => MinOrMaxFusion::ToMin,
        (vc::CompareOp::Lt | vc::CompareOp::Le, true) => MinOrMaxFusion::ToMin,
        (vc::CompareOp::Lt | vc::CompareOp::Le, false) => MinOrMaxFusion::ToMax,
        (vc::CompareOp::Eq | vc::CompareOp::Neq, _) => {
            todo!(
                "Unable to lower compare and select into max/min operation (VectorChainHelper.cpp:463)"
            )
        }
    }
}

/// ARE TWO OPS THE SAME COMPUTATION? — `dcc::OperationEquivalence::operationsAreEquivalent`
/// (`dcc/src/Analysis/OperationEquivalence.cpp:88-334`), at the constructor defaults
/// [`e065_fuseCompareAndSelectIntoMinOrMax`](fuse_compare_and_select_into_min_or_max) uses.
///
/// ⛔ NOT AN ANCHORED UNIT — it is `dcc/src/Analysis/`, not this campaign's file list, and it is here
/// because e065 is unportable without it (the `relu.mlir` duplicate-constant case above). The
/// defaults it is built with are `dcc::OperationEquivalence oe;`, i.e.
/// `do_recursive_compare = true, all_block_args_are_equiv = true, use_equiv_classes = true`
/// (`OperationEquivalence.hpp:27-34`), with a null `operands_equiv_checker`.
///
/// ⭐ `preference` IS `functor_` TOGETHER WITH ITS `context_`, and it is `HighPreference::None` for
/// e065 and every other site but one. Entry 193 is the exception, and it is a second caller of this
/// function: see [`HighPreference::prefers`].
///
/// ⛔ `block_args` IS A PARAMETER AND NOT A DEFAULT, because one site overrides it: every conditional
/// tree constructs its comparison with `/*all_block_args_are_equiv*/ false`
/// (`CFGSDataflowConditionalTree.hpp:112-115`, entry 100), and entry 284 compares two `scf.if`s whose
/// conditions are region arguments. Baking in the constructor default would merge two loops.
///
/// The reference's shape, in order: same pointer ⇒ true; dialect, name, result count, operand count,
/// region count, successor count and result types must match; attribute dictionaries compared
/// pairwise with `dbgName` filtered out and `AffineMapAttr`/`IntegerSetAttr` given value comparisons;
/// then the operand list; then the regions.
///
/// ⭐⭐ EVERYTHING BEFORE THE OPERAND LOOP IS ONE `==` HERE, because in this island an op's dialect,
/// name, arity, attributes and result types are exactly the parts of its variant that are *not*
/// [`Val`]s or nested ops. [`skeleton`] blanks those two and compares what is left — so a `Vec` of
/// operands still compares by length (arity), a `reduction_map` still compares as an
/// [`AffineMap`](crate::islands::dataflow_ir::ty::AffineMap), and no hand-written per-variant
/// comparison can fall behind the enum.
///
/// ⚠️ THREE DELIBERATE DIVERGENCES, none of which can change an answer:
/// - **The `dbgName` filter has nothing to filter.** No `dataflow_ir` op in this island carries a
///   debug name — the reference's names are a printing concern this rung does not model — so the
///   filtered dictionaries are the full ones.
/// - **`IntegerSetAttr` gets order-insensitive constraint matching in the reference and plain
///   equality here.** Ours are built by [`IntegerSet::from_sizes`], one constraint per dimension in
///   dimension order, so two equal sets are equal element-wise.
/// - **The equivalence-class memo (`use_equiv_classes_`) is omitted.** `eq_classes_.unionSets` only
///   short-circuits a *repeat* question; it cannot answer one differently.
///
/// ⛔ AND ONE PLACE THE REFERENCE HAS DEAD CODE. When two differing operands share a defining op it
/// compares their result numbers and calls them identical if equal — but two *different* values from
/// one op have different result numbers by construction, so that branch never fires and a shared
/// definition is always a mismatch.
pub(crate) fn ops_are_equivalent(
    a: &DfirOp,
    b: &DfirOp,
    scope: &[DfirOp],
    preference: HighPreference,
    block_args: BlockArgEquivalence,
) -> bool {
    // `if (&op_a == &op_b) return true;` — also the recursion's cycle guard.
    if core::ptr::eq(a, b) {
        return true;
    }

    // `if (functor_ && functor_(op_a, op_b, context_)) return true;` (`:115-118`), which sits after
    // the equivalence-class check this port drops and before every structural test below it.
    if preference.prefers(a, b) {
        return true;
    }

    if skeleton(a) != skeleton(b) {
        return false;
    }

    // The operand loop. `zip` is the reference's own `llvm::zip`; the counts already agree.
    for (read_a, read_b) in dfir_op::operands(a).into_iter().zip(dfir_op::operands(b)) {
        if read_a == read_b {
            continue;
        }
        match (
            dfir_op::defining_op(read_a, scope),
            dfir_op::defining_op(read_b, scope),
        ) {
            // Both are results. One shared definition means differing result numbers, hence a
            // mismatch; otherwise ask whether the two definitions compute the same thing.
            (Some(def_a), Some(def_b)) => {
                if core::ptr::eq(def_a, def_b)
                    || !ops_are_equivalent(def_a, def_b, scope, preference, block_args)
                {
                    return false;
                }
            }
            // Both are region arguments. `all_block_args_are_equiv_` accepts any pair (`:292-293`);
            // ⛔ `false` DEMANDS THE SAME ARGUMENT OF THE SAME OWNER (`:294-300`) AND NO LOOKUP CAN
            // SAY YES HERE: two distinct [`Val`]s are never one argument of one region, and the equal
            // pair was taken by `read_a == read_b` above.
            (None, None) => match block_args {
                BlockArgEquivalence::AllEquivalent => {}
                BlockArgEquivalence::SameOwnerAndIndex => return false,
            },
            // One of each is a mismatch.
            (Some(_), None) | (None, Some(_)) => return false,
        }
    }

    // `do_recursive_compare_` is true: descend. `regionsAreEquivalent` compares region counts (equal
    // by variant here) and then block sizes, which is this length check.
    for (region_a, region_b) in dfir_op::regions(a).into_iter().zip(dfir_op::regions(b)) {
        if region_a.len() != region_b.len() {
            return false;
        }
        for (inner_a, inner_b) in region_a.iter().zip(region_b.iter()) {
            if !ops_are_equivalent(inner_a, inner_b, scope, preference, block_args) {
                return false;
            }
        }
    }

    true
}

/// ONE OP WITH EVERY SSA NAME AND EVERY NESTED OP REMOVED — what [`ops_are_equivalent`] compares
/// before it looks at operands.
///
/// ⛔ NOT A GENERAL-PURPOSE OPERATION, which is why it is private and returns a value nobody may
/// emit: blanking the results leaves an op that binds `%0` twice. It exists to turn "same dialect,
/// name, arity, attributes and result types" into one `==`.
fn skeleton(op: &DfirOp) -> DfirOp {
    let mut bare = op.clone();
    for region in dfir_op::regions_mut(&mut bare) {
        region.clear();
    }
    for (_, val) in dfir_op::vals_mut(&mut bare) {
        *val = Val(0);
    }
    bare
}

/// Replaces: e066_resetSentientFMAsIfExists
///
/// **066/384** `vectorchain::resetSentientFMAsIfExists` —
/// `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:468` (13L).
///
/// ```cpp
/// void vectorchain::resetSentientFMAsIfExists(dataflow::ProgramUnitOp unit) {
///   Builder builder(unit);
///   auto attr = builder.getSI32IntegerAttr(-1);
///   unit.walk<WalkOrder::PreOrder>([&](mlir::Operation* op) {
///     if (auto mac_op = llvm::dyn_cast<sentient::MacOp>(op)) {
///       mac_op.setOpADataIDAttr(attr);
///       mac_op.setOpBDataIDAttr(attr);
///       mac_op.setOpCDataIDAttr(attr);
///     } else if (auto bin_op = llvm::dyn_cast<sentient::BinaryOp>(op)) {
///       bin_op.setOpADataIDAttr(attr);
///       bin_op.setOpBDataIDAttr(attr);
///     }
///   });
/// }
/// ```
///
/// ⭐⭐ THE `-1` IS "UNASSIGNED", NOT "DATA ID MINUS ONE", so this is
/// [`Operand::data_id`](sen::Operand::data_id)` = None` on every compute the unit already carries.
/// It runs before `OperandReuse` re-derives them
/// (`VectorChainToSentientPT.cpp:1006-1007`, `VectorChainToSentientPESFP.cpp:1385-1389`, both under
/// the reference's own *"Order of these operations is important!"*): a unit lowered twice would
/// otherwise keep the first pass's ids and reuse would be computed against stale numbering.
///
/// ⛔⛔ MAC AND BINARY ONLY — NOT UNARY, NOT TERNARY, AND THAT IS THE REFERENCE'S `else if` CHAIN, NOT
/// AN OMISSION HERE. `sentient::UnaryOp` and `sentient::TernaryOp` are their own op classes that
/// neither `dyn_cast` matches, so [`sen::Op::VectorUnary`] and [`sen::Op::VectorTernary`] keep the
/// data ids they were emitted with.
///
/// ⛔ THE WALK IS PROVABLY COMPLETE WITH THREE RECURSION ARMS. Only [`sen::Op::For`] and
/// [`sen::Op::If`] hold a `Vec<`[`SenOp`]`>`; every other nested body in this island is a
/// `Vec<`[`DfirOp`]`>`, a type with no sentient variant at all, so no compute can hide inside one.
/// And the match is total over both enums: a twenty-ninth `sentient` op that carries an
/// [`Operand`](sen::Operand) stops the build here rather than silently keeping a stale id.
pub fn reset_sentient_fmas_if_exists<A: Arch>(unit: &mut SentientProgramUnit<A>) {
    reset_data_ids(&mut unit.body);
}

/// The `unit.walk<WalkOrder::PreOrder>` of [`reset_sentient_fmas_if_exists`], over one block.
fn reset_data_ids(ops: &mut [SenOp]) {
    for op in ops {
        match op {
            SenOp::Sentient(sentient_op) => match sentient_op {
                sen::Op::VectorMac {
                    op_a, op_b, op_c, ..
                } => {
                    op_a.data_id = None;
                    op_b.data_id = None;
                    op_c.data_id = None;
                }
                sen::Op::VectorBinary { op_a, op_b, .. } => {
                    op_a.data_id = None;
                    op_b.data_id = None;
                }
                sen::Op::For { body, .. } => reset_data_ids(body),
                sen::Op::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    reset_data_ids(then_body);
                    reset_data_ids(else_body);
                }
                sen::Op::Yield { .. }
                | sen::Op::VectorUnary { .. }
                | sen::Op::VectorTernary { .. }
                | sen::Op::Load { .. }
                | sen::Op::LoadAndSend { .. }
                | sen::Op::ReceiveAndStore { .. }
                | sen::Op::LoadAndStore { .. }
                | sen::Op::LoadComputeAndSend { .. }
                | sen::Op::LoadAndExtractScalar { .. }
                | sen::Op::ReceiveAndExtractScalar { .. }
                | sen::Op::ScalarAdd { .. }
                | sen::Op::ScalarSub { .. }
                | sen::Op::ScalarMul { .. }
                | sen::Op::ScalarCopy { .. }
                | sen::Op::ScalarConstant { .. }
                | sen::Op::VectorConstant { .. }
                | sen::Op::Sync { .. }
                | sen::Op::Nop { .. }
                | sen::Op::SetSendDst { .. }
                | sen::Op::LogicalPort { .. }
                | sen::Op::Splat { .. }
                | sen::Op::Samv { .. }
                | sen::Op::SetMask { .. }
                | sen::Op::IncrMask { .. }
                | sen::Op::Opaque { .. } => {}
            },
            // A time loop's body IS at this rung, so it is walked; the other shared dialects' nested
            // bodies are `Vec<DfirOp>` — no compute can be in one — and `symbol.create_symbol` has
            // no body at all.
            SenOp::AffineFor(loop_op) => reset_data_ids(&mut loop_op.body),
            SenOp::Dataflow(_)
            | SenOp::Agen(_)
            | SenOp::VectorChain(_)
            | SenOp::Affine(_)
            | SenOp::Vector(_)
            | SenOp::Arith(_)
            | SenOp::Scf(_)
            | SenOp::Symbol(_)
            | SenOp::Uniform(_) => {}
        }
    }
}

/// Replaces: e067_redefineConstantVectors
///
/// **067/384** `vectorchain::redefineConstantVectors` —
/// `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:571` (37L).
///
/// ```cpp
/// // The purpose of this method is to undo the effects of "SCCP" or
/// // Canonicalizer pass in grouping multiple constant vectors with same values
/// // into a single vector.
/// // Reusing same SSA variable of constant vector operation for multiple
/// // different vector operations breaks some assumptions leading to
/// // incorrect lowering.
/// void vectorchain::redefineConstantVectors(Operation* op) {
///   struct updateInfo {
///     Operation* op_to_update_;
///     Value update_with_;
///     unsigned int location_;
///   };
///
///   std::vector<updateInfo> info;
///   std::vector<Operation*> to_be_deleted;
///   op->walk<WalkOrder::PreOrder>([&](arith::ConstantOp const_op) {
///     auto is_vec = mlir::isa<VectorType, dataflow::CustomVectorType>(
///         const_op.getResult().getType());
///     if (is_vec && !const_op->use_empty()) {
///       // MLIR doesn't allow updating uses while operating on getUses
///       // If try instead, we are getting incorrect set of uses.
///       for (auto& use : const_op->getUses()) {
///         auto* owner = use.getOwner();
///         OpBuilder builder(owner);
///         auto* new_op = builder.clone(*const_op);
///         info.push_back({owner, new_op->getResult(0), use.getOperandNumber()});
///       }
///
///       to_be_deleted.push_back(const_op);
///     }
///   });
///
///   for (auto& group : info) {
///     group.op_to_update_->setOperand(group.location_, group.update_with_);
///   }
///
///   for (auto* tmp_op : to_be_deleted) {
///     tmp_op->dropAllUses();
///     tmp_op->erase();
///   }
/// }
/// ```
///
/// ⭐⭐ ONE VECTOR CONSTANT PER USE, EACH IMMEDIATELY BEFORE ITS USER. `const-vector-multiple-uses.mlir`
/// is the vendor case named after it: three `arith.constant dense<…>` ops at the top of the function
/// (`:179`, `:184`, `:185`), `%cst_1` read six times from four different loop depths — and in the
/// expectation (`:15-24`) the function head keeps `arith.constant true`, `false` and the index
/// constants and has *no* `dense<…>` left. Every one was replaced by a clone next to its reader, and
/// the original erased.
///
/// ⛔⛔ WHY IT MATTERS, IN THE REFERENCE'S OWN WORDS: this undoes SCCP/Canonicalizer CSE-ing equal
/// vector constants into one SSA value, because *"reusing same SSA variable of constant vector
/// operation for multiple different vector operations breaks some assumptions leading to incorrect
/// lowering"*. The lowering that follows attaches per-*use* facts to a constant — which compute port
/// it splats to, which precision, which data id — so two computes sharing one constant value fight
/// over one op's attributes.
///
/// ⭐ POSITIONAL REWRITING, NOT `substitute(from, to)`. The reference clones once per **use**, so an
/// op reading the same constant in two operand slots gets two distinct clones. A value-for-value
/// substitution would give it one and re-create exactly the sharing this function exists to undo.
/// [`dfir_op::vals_mut`] is what makes the per-slot assignment expressible.
///
/// ⛔ AND IT FILTERS [`dfir_op::vals_mut`] ITSELF RATHER THAN CALLING [`dfir_op::operands_mut`],
/// WHICH IS NARROWER ON PURPOSE. That one withholds a link end and a [`vc::Predicate`] because
/// re-*pointing* either alone would break its pairing with something else; `getUses()` withholds
/// nothing, and cloning cannot break either pairing — a link end names a unit and never a constant,
/// and the clone carries the SAME `Vector` type as the constant it copies, so a mask keeps its width.
///
/// ⛔ `!use_empty()` IS A REAL GUARD, NOT AN OPTIMISATION: an unused vector constant is neither cloned
/// nor erased, and stays exactly where it was.
///
/// ⭐ `isa<VectorType, dataflow::CustomVectorType>` COLLAPSES TO A VARIANT TEST HERE.
/// [`arith::Op::DenseConstant`](dfir_op::arith::Op::DenseConstant) is this island's only vector-typed
/// constant; [`Constant`](dfir_op::arith::Op::Constant) and
/// [`ConstantInt`](dfir_op::arith::Op::ConstantInt) are scalars, which is why the vendor
/// expectation keeps its `arith.constant true` and its indices.
///
/// ⚠️ `values` IS THE ONE ARGUMENT THE REFERENCE DOES NOT HAVE, and it is the permitted kind of
/// divergence: `builder.clone` mints an SSA name from the MLIR context, and in this island names come
/// from [`Values`]. The reference walks a whole `module_op`
/// (`VectorChainToSentientPT.cpp:983`), which here is [`Program::preamble`] plus every unit body —
/// hence [`ProgramUnits::iter_mut`](crate::islands::dataflow_ir::ProgramUnits::iter_mut).
pub fn redefine_constant_vectors<A: Arch>(program: &mut DfirProgram<A>, values: &mut Values) {
    // The `walk<WalkOrder::PreOrder>([](arith::ConstantOp))` half: which vector constants exist, and
    // the op to clone for each. Keyed by the value it binds, which is how a use names it.
    let mut templates: BTreeMap<Val, DfirOp> = BTreeMap::new();
    collect_vector_constants(&program.preamble, &mut templates);
    for unit in program.units.iter() {
        collect_vector_constants(&unit.body, &mut templates);
    }

    // `!const_op->use_empty()`.
    templates.retain(|bound, _| {
        !dfir_op::uses(*bound, &program.preamble).is_empty()
            || program
                .units
                .iter()
                .any(|unit| !dfir_op::uses(*bound, &unit.body).is_empty())
    });

    // The two rewrite loops, fused: `setOperand` per use with a clone placed before the user, and the
    // original erased. One pass suffices because the clones are fresh values nothing else reads, so
    // there is no "updating uses while operating on getUses" hazard to sequence around.
    clone_constants_per_use(&mut program.preamble, &templates, values);
    for unit in program.units.iter_mut() {
        clone_constants_per_use(&mut unit.body, &templates, values);
    }
}

/// Every `arith.constant` of vector type in `block` and below, by the value it binds.
fn collect_vector_constants(block: &[DfirOp], into: &mut BTreeMap<Val, DfirOp>) {
    for op in block {
        if let DfirOp::Arith(dfir_op::arith::Op::DenseConstant { result, .. }) = op {
            into.insert(*result, op.clone());
        }
        for region in dfir_op::regions(op) {
            collect_vector_constants(region, into);
        }
    }
}

/// Rebuild one block with a private clone of every tracked constant in front of each op that reads
/// it, and the originals dropped.
fn clone_constants_per_use(
    block: &mut Vec<DfirOp>,
    templates: &BTreeMap<Val, DfirOp>,
    values: &mut Values,
) {
    let mut rebuilt: Vec<DfirOp> = Vec::with_capacity(block.len());
    for mut op in block.drain(..) {
        // Descend first: a use inside a region gets its clone inside that region, which is what
        // `OpBuilder builder(owner)` does when the owner is nested.
        for region in dfir_op::regions_mut(&mut op) {
            clone_constants_per_use(region, templates, values);
        }

        // `tmp_op->dropAllUses(); tmp_op->erase();` — by the time this runs no use is left.
        if let DfirOp::Arith(dfir_op::arith::Op::DenseConstant { result, .. }) = &op
            && templates.contains_key(result)
        {
            continue;
        }

        let mut clones: Vec<DfirOp> = Vec::new();
        for slot in dfir_op::vals_mut(&mut op)
            .into_iter()
            .filter_map(|(role, val)| (role == dfir_op::Role::Operand).then_some(val))
        {
            if let Some(template) = templates.get(slot) {
                let fresh = values.mint();
                let mut clone = template.clone();
                for (role, val) in dfir_op::vals_mut(&mut clone) {
                    if role == dfir_op::Role::Result {
                        *val = fresh;
                    }
                }
                clones.push(clone);
                *slot = fresh;
            }
        }
        rebuilt.extend(clones);
        rebuilt.push(op);
    }
    *block = rebuilt;
}

/// Replaces: e068_getVectorBinaryToSentientBinary
///
/// **068/384** `getVectorBinaryToSentientBinary` —
/// `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:32` (17L).
///
/// ```cpp
/// static SentientBinaryOperator getVectorBinaryToSentientBinary(
///     VectorChainBinaryOperator vb) {
///   if (vb == VectorChainBinaryOperator::and0)     return SentientBinaryOperator::and0;
///   if (vb == VectorChainBinaryOperator::or0)      return SentientBinaryOperator::or0;
///   if (vb == VectorChainBinaryOperator::xnor)     return SentientBinaryOperator::xnor;
///   if (vb == VectorChainBinaryOperator::and_not)  return SentientBinaryOperator::and_not;
///   if (vb == VectorChainBinaryOperator::min)      return SentientBinaryOperator::min;
///   if (vb == VectorChainBinaryOperator::max)      return SentientBinaryOperator::max;
///   if (vb == VectorChainBinaryOperator::abs_min)  return SentientBinaryOperator::abs_min;
///   if (vb == VectorChainBinaryOperator::abs_max)  return SentientBinaryOperator::abs_max;
///   if (vb == VectorChainBinaryOperator::add)      return SentientBinaryOperator::add;
///   if (vb == VectorChainBinaryOperator::mul)      return SentientBinaryOperator::mul;
///   if (vb == VectorChainBinaryOperator::mul_div2) return SentientBinaryOperator::mul_div2;
///   if (vb == VectorChainBinaryOperator::sub)      return SentientBinaryOperator::sub;
///
///   DT_ERROR("unknown vectorchain binary operator");
/// }
/// ```
///
/// ⭐⭐ TWELVE IN, TWELVE OUT, AND THE `DT_ERROR` IS UNREACHABLE — so this port carries no `todo!`.
/// [`vc::BinaryOp`] has exactly twelve variants and all twelve appear above; the reference needs the
/// trap only because a C++ enum argument can hold a value no enumerator names.
///
/// ⛔ IT IS NOT AN IDENTITY EVEN THOUGH IT LOOKS LIKE ONE. [`sen::BinaryOp`] has twenty variants —
/// the converts, `merge`, `pack` and the four `fcmp`s have no `vectorchain` counterpart — so the two
/// enums are genuinely different sets and this is the map between them.
#[must_use]
pub const fn vector_binary_to_sentient_binary(vb: vc::BinaryOp) -> sen::BinaryOp {
    match vb {
        vc::BinaryOp::And => sen::BinaryOp::And,
        vc::BinaryOp::Or => sen::BinaryOp::Or,
        vc::BinaryOp::Xnor => sen::BinaryOp::Xnor,
        vc::BinaryOp::AndNot => sen::BinaryOp::AndNot,
        vc::BinaryOp::Min => sen::BinaryOp::Min,
        vc::BinaryOp::Max => sen::BinaryOp::Max,
        vc::BinaryOp::AbsMin => sen::BinaryOp::AbsMin,
        vc::BinaryOp::AbsMax => sen::BinaryOp::AbsMax,
        vc::BinaryOp::Add => sen::BinaryOp::Add,
        vc::BinaryOp::Mul => sen::BinaryOp::Mul,
        vc::BinaryOp::MulDiv2 => sen::BinaryOp::MulDiv2,
        vc::BinaryOp::Sub => sen::BinaryOp::Sub,
    }
}

/// Replaces: e069_getVectorElementWiseCompareOperatorToSentientBinaryOperator
///
/// **069/384** `getVectorElementWiseCompareOperatorToSentientBinaryOperator` —
/// `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:55` (9L).
///
/// ```cpp
/// static SentientBinaryOperator
/// getVectorElementWiseCompareOperatorToSentientBinaryOperator(
///     VectorChainElementWiseCompareOperator vb) {
///   if (vb == VectorChainElementWiseCompareOperator::compare_eq)  return SentientBinaryOperator::fcmp_eq;
///   if (vb == VectorChainElementWiseCompareOperator::compare_neq) return SentientBinaryOperator::fcmp_neq;
///   if (vb == VectorChainElementWiseCompareOperator::compare_lt)  return SentientBinaryOperator::fcmp_lt;
///   if (vb == VectorChainElementWiseCompareOperator::compare_le)  return SentientBinaryOperator::fcmp_le;
///
///   DT_ERROR("unknown vectorchain Ternary operator");
/// }
/// ```
///
/// ⭐⭐ FOUR ARMS FOR SIX INPUTS, AND `compare_gt`/`compare_ge` ARE ABSENT ON PURPOSE — THE ISA HAS NO
/// GREATER-THAN. The caller reorders instead: *"Since Sentient ISA & Dialect doesn't support
/// element-wise gt, ge, we reorder the input operands before lowering to Sentient IR"*
/// (`VectorChainToSentientPESFP.cpp:450-485`), so a `compare_gt` arrives here already flipped to
/// `compare_lt`. `fcmp_select.mlir` shows both halves of that: input `:136` is
/// `element_wise_compare %53, %55 {compare_gt}` and the expectation `:48` is
/// `binaryOp = #sentient<binary_operator fcmp_lt>` with `opA = lrf1` (`%55`) before `opB = lx`
/// (`%53`) — the operands swapped.
///
/// ⛔ SO THE `todo!` IS REACHED ONLY BY A CALLER THAT SKIPPED THAT REORDERING, which is a defect in
/// the caller and not an unported operator. ⛔ AND ITS MESSAGE SAYS "Ternary" — the reference's own
/// copy-paste from `getVectorTernaryToSentientTernary`, kept verbatim so a grep for the emitted text
/// finds the C++ line.
#[must_use]
pub fn vector_element_wise_compare_operator_to_sentient_binary_operator(
    vb: vc::CompareOp,
) -> sen::BinaryOp {
    match vb {
        vc::CompareOp::Eq => sen::BinaryOp::CompareEq,
        vc::CompareOp::Neq => sen::BinaryOp::CompareNeq,
        vc::CompareOp::Lt => sen::BinaryOp::CompareLt,
        vc::CompareOp::Le => sen::BinaryOp::CompareLe,
        vc::CompareOp::Gt | vc::CompareOp::Ge => todo!(
            "unknown vectorchain Ternary operator — gt/ge must be reordered into lt/le first (VectorChainHelper.hpp:64, VectorChainToSentientPESFP.cpp:450-485)"
        ),
    }
}

/// Replaces: e070_getVectorTernaryToSentientTernary
///
/// **070/384** `getVectorTernaryToSentientTernary` —
/// `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:247` (7L).
///
/// ```cpp
/// static SentientTernaryOperator getVectorTernaryToSentientTernary(
///     Operation* op) {
///   if (isa<vectorchain::ElementWiseSelectionOp>(op)) {
///     return SentientTernaryOperator::select;
///   }
///
///   DT_ERROR("unknown vectorchain Ternary operator");
/// }
/// ```
///
/// ⭐ ONE TERNARY EXISTS AND IT IS `select`. `fcmp_select.mlir:156` is the
/// `vectorchain.element_wise_selection` and `:64` is its
/// `ternaryOp = #sentient<ternary_operator select>`.
///
/// ⛔ THE ARGUMENT NARROWS FROM `Operation*` TO [`&vc::Op`](vc::Op) — a non-`vectorchain` op fails
/// `isa<ElementWiseSelectionOp>` and reaches the `DT_ERROR` anyway, so nothing an outer dialect could
/// pass has an answer this function could give. The remaining seventeen `vectorchain` ops keep it —
/// [`Select`](vc::Op::Select) among them, which despite the name is the reference's own
/// `vectorchain.select` and NOT an `ElementWiseSelectionOp`, so the C++ falls through on it too.
#[must_use]
pub fn vector_ternary_to_sentient_ternary(op: &vc::Op) -> sen::TernaryOp {
    match op {
        vc::Op::ElementWiseSelection { .. } => sen::TernaryOp::Select,
        vc::Op::Estimate { .. }
        | vc::Op::FastExp { .. }
        | vc::Op::Floor { .. }
        | vc::Op::Neg { .. }
        | vc::Op::ScanWithGap { .. }
        | vc::Op::Select { .. }
        | vc::Op::Multiply { .. }
        | vc::Op::MultiplyAccumulate { .. }
        | vc::Op::ElementWiseCompare { .. }
        | vc::Op::Binary { .. }
        | vc::Op::Pack { .. }
        | vc::Op::Merge { .. }
        | vc::Op::ConstantBitstream { .. }
        | vc::Op::Shuffle { .. }
        | vc::Op::Rotate { .. }
        | vc::Op::Cast { .. }
        | vc::Op::CreateAffineMask { .. }
        | vc::Op::CreateAffineMaskSet { .. } => {
            todo!("unknown vectorchain Ternary operator (VectorChainHelper.hpp:254)")
        }
    }
}

// ═══════════════════════════════════════════ 163/384 ═══════════════════════════════════════════

/// WHICH SLICES OF A STICK A NON-PT MASK TURNS OFF — the `unsigned` `getMaskValueConstantForNonPT`
/// answers, as one bit per slice.
///
/// ⛔⛔ A BITMAP, AND THE PT'S MASK VALUE IS A COUNT. `getMaskValueForPT` returns
/// `masked_columns` — a NUMBER of columns, carried by
/// [`super::vc_loop_mask_tree::MaskedColumns`] — while this side sums `1 << i` over a RANGE OF SLICE INDICES
/// (`VectorChainHelper.cpp:130-131`). Both end up in the same
/// `sentient.scalar_constant {value = N : si64}` attribute, so a bare integer would let the two be
/// interchanged silently: `2` means *"two columns are masked"* on the PT and *"slice 1 is masked"*
/// here.
///
/// ⭐ AND IT IS AT MOST [`Arch::SLICES_PER_STICK`] BITS WIDE, which is what makes the shift in
/// [`get_mask_value_constant_for_non_pt`] total.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SliceMask(u32);

impl SliceMask {
    /// The literal the mask constant carries.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }
}

/// Replaces: e163_getMaskValueConstantForNonPT
///
/// # THE MASK VALUE A PE, SFP OR L-UNIT COMPUTE TAKES, READ OFF THE `create_affine_mask`
///
/// ```cpp
/// std::optional<unsigned> vectorchain::getMaskValueConstantForNonPT(
///     Operation* op, Operation* operand, const SenSystemDef& sys_def) {
///   if (!operand->hasAttr("mask_set")) {
///     operand->emitOpError("Mask operation doesn't have mask_set attribute.");
///     return std::nullopt;
///   }
///   auto mask_set_attr = operand->getAttr("mask_set");
///   auto mask_set = cast<IntegerSetAttr>(mask_set_attr).getValue();
///   if (mask_set.getNumDims() != 1) {
///     operand->emitOpError("Mask affine set has to have one dimension.");
///     return std::nullopt;
///   }
///   if (mask_set.getNumSymbols() > 0) {
///     operand->emitOpError("Mask affine set should not have any symbols");
///     return std::nullopt;
///   }
///   affine::FlatAffineValueConstraints mask_set_flat(mask_set);
///   if (!hasConstantBounds(mask_set_flat)) {
///     operand->emitOpError("Mask affine set has to have constant bounds.");
///     return std::nullopt;
///   }
///   auto lowerbound_elm = int64_t(mask_set_flat.getConstantBound(BoundType::LB, 0).value());
///   auto upperbound_elm = int64_t(mask_set_flat.getConstantBound(BoundType::UB, 0).value());
///   DT_CHECK_MSG(isa<VectorType>(operand->getOpResult(0).getType()),
///                "expecting operand to be VectorType");
///   auto dim = cast<VectorType>(operand->getOpResult(0).getType()).getDimSize(0);
///   DT_CHECK(dim / sys_def.numSlicesPerStick > 0);
///   unsigned from_slice = lowerbound_elm / (dim / sys_def.numSlicesPerStick) + 1;
///   unsigned to_slice = upperbound_elm / (dim / sys_def.numSlicesPerStick);
///   unsigned mask_val = 0;
///   for (unsigned i = from_slice - 1; i <= to_slice; i++) mask_val += 1 << i;
///   return mask_val;
/// }
/// ```
/// (`dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:93-133`)
///
/// # ⭐⭐ THE `+ 1` AND THE `- 1` CANCEL, AND THE ANSWER IS A CLOSED RANGE OF SLICES
///
/// `from_slice` is `lb / lanes_per_slice + 1` and the loop starts at `from_slice - 1`, so the mask is
/// exactly `OR of 1 << i for i in (lb / lanes_per_slice) ..= (ub / lanes_per_slice)` — the slices the
/// masked span touches, with `lanes_per_slice = dim / numSlicesPerStick`. ⛔ THE ROUND TRIP IS NOT
/// COSMETIC, THOUGH: it happens through an `unsigned`, which is where the negative case below comes
/// from.
///
/// # ⛔ IT EMITS NOTHING, AND THAT IS THE SPLIT THE REFERENCE MADE
///
/// The `sentient.scalar_constant` this value becomes is emitted one level up, by the header template
/// `getMaskValueForNonPT` (entry 229) — `builder.setInsertionPoint(op)` then
/// `sentient::ConstantOp::create(builder, op->getLoc(), builder.getIndexType(), mask_val.value())`
/// (`VectorChainHelper.hpp:114-125`). So this entry legitimately returns a number: the op belongs to
/// the entry that has the builder.
///
/// # ⛔ THE MASK **PARAMETER** IS IGNORED, WHERE THE PT FOLDS IT IN
///
/// This side reads the `mask_set` attribute and only that. A `create_affine_mask %14 {mask_set}` on
/// a PE therefore lowers as though it were static, and a set that actually USES its symbol is refused
/// outright by `getNumSymbols() > 0`. The PT sibling instead folds a constant parameter into the set
/// and accepts a loop-dependent one (entry 089) — two masks with the same spelling and two different
/// readings, which is the reference's own asymmetry and not a simplification here.
///
/// ⭐ AND THE FIRST PARAMETER IS UNUSED. `op` is the masked compute; nothing in the body touches it.
/// It exists for the diagnostics — `operand->emitOpError` is called on the MASK — and for the
/// template above, which needs it for the insertion point.
///
/// # THE GOLDEN, AND WHY IT IS ALWAYS ZERO
///
/// `#set = affine_set<(d0) : (d0 - 64 >= 0, -d0 + 63 >= 0)>` over `vector<64xi1>`
/// (`dcc/test/Conversion/VectorChainToSentientPESFP/fnms_with_cast.mlir:20`, `:30`): `Lb` is 64 and
/// `Ub` is 63, `lanes_per_slice` is 8, so the range is `8 ..= 7` — EMPTY — and the mask is 0. The
/// `CHECK-SENT-IR` is `sentient.scalar_constant {value = 0 : si64}` feeding
/// `vector_mac mask(...)` (`:11-12`).
///
/// ⭐ A CENSUS, NOT A SAMPLE: every one-dimensional `mask_set` a `create_affine_mask` carries in the
/// authority's 825 `.mlir` files is of that all-lanes-live shape — `(d0 - N >= 0, -d0 + (N-1) >= 0)`
/// for N of 64, 128, 256 and 2048 — so every non-PT mask constant IBM's own tests produce is 0. The
/// non-zero answers below are derived from the arithmetic and marked as such.
///
/// ⛔ SO `from_slice` MUST NOT BE RANGE-CHECKED AT ITS TOP END. 64 / 8 = 8 is one past the last
/// slice of the stick, and refusing it would refuse every mask in the corpus. The empty range is the
/// answer.
///
/// # THE STOPS, EACH A CITED ABORT OR UB
///
/// * `DT_CHECK(dim / sys_def.numSlicesPerStick > 0)` — a mask narrower than a stick has slices
///   (`:127`). Declined here, in the crate's manner: `DT_CHECK` throws
///   (`util/dt_exception.hpp:110-121`), and a vector of fewer than eight lanes is not a mask this
///   lowering can read either way.
/// * ⛔ A NEGATIVE `lb / lanes_per_slice` IS AN UNSIGNED WRAP THERE, AND THE MASK IS 0. `from_slice`
///   is an `unsigned`: `lb / lps == -1` makes it 0 and `from_slice - 1` becomes `0xFFFFFFFF`, and
///   anything below that makes it larger still — the loop condition is false at once either way.
///   This port answers 0 for the same input, which reproduces the OBSERVABLE result rather than the
///   wrap.
/// * ⛔ AN UPPER SLICE PAST THE STICK IS `1 << i` OUT OF RANGE, AND IS DECLINED. There is no upper
///   guard on this side — the PT's sibling checks `upper_bound == op_num_elems - 1` explicitly
///   (`VectorChainToSentientPT/Helper.cpp:96-99`) and this one checks nothing — so a `dim` that is
///   not a multiple of eight (say 12, giving `lps == 1` and `to_slice == 11`) shifts past the
///   eight-bit slice mask, and a `dim` of 32 or less with a wide span shifts past 31, which is
///   undefined behaviour in C++. `to_slice` outside `0 .. SLICES_PER_STICK` is refused here; ⭐ no
///   mask in the corpus is affected, because every one of them is stated over a multiple of eight
///   lanes.
///
/// # THE DT_CHECK THAT IS **DISCHARGED BY CONSTRUCTION**
///
/// `isa<VectorType>(operand->getOpResult(0).getType())` (`:123-124`) — both mask forms of this island
/// carry a [`Vector`], so the mask's result type cannot be anything else and `getDimSize(0)` cannot
/// be asked of a scalar.
#[must_use]
pub fn get_mask_value_constant_for_non_pt<A: Arch>(mask: &vc::Op) -> Option<SliceMask> {
    // `if (!operand->hasAttr("mask_set"))` — ⭐ THE TWO ISLAND VARIANTS ARE ONE C++ OP, and both
    // carry the attribute; the prefix form's set is written by [`vc::LaneMask::as_set`]. Any other op
    // has no `mask_set` at all, which is the first refusal.
    let (mask_set, mask_ty) = match mask {
        vc::Op::CreateAffineMaskSet { mask_set, ty, .. } => (mask_set.clone(), *ty),
        vc::Op::CreateAffineMask { mask, .. } => (mask.as_set()?, mask.ty()),
        // ⭐ A WITNESS MATCH, not a walk — the precedent [`super::vc_helper::get_mask_value_for_pt`]
        // set for the same pair of variants.
        _ => return None,
    };

    // `if (mask_set.getNumDims() != 1)` — "Mask affine set has to have one dimension."
    if mask_set.dims != 1 {
        return None;
    }

    // `if (mask_set.getNumSymbols() > 0)` — "Mask affine set should not have any symbols". ⛔ THIS IS
    // WHERE A DYNAMIC MASK STOPS on this side; see the note on the mask parameter above.
    if mask_set.symbols > 0 {
        return None;
    }

    // `if (!hasConstantBounds(mask_set_flat))` — entry 063, whose only caller is this line.
    if !has_constant_bounds(&mask_set) {
        return None;
    }

    // The two `.value()` calls. [`has_constant_bounds`] above is what makes both present, so the `?`
    // restates that gate rather than adding a second refusal.
    let lowerbound_elm = mask_set.constant_bound(BoundType::Lb, 0)?;
    let upperbound_elm = mask_set.constant_bound(BoundType::Ub, 0)?;

    // `auto dim = cast<VectorType>(operand->getOpResult(0).getType()).getDimSize(0);`
    let dim = i64::try_from(mask_ty.len).ok()?;
    let slices_per_stick = i64::from(A::SLICES_PER_STICK);

    // `DT_CHECK(dim / sys_def.numSlicesPerStick > 0);`
    let lanes_per_slice = dim / slices_per_stick;
    if lanes_per_slice <= 0 {
        return None;
    }

    // `from_slice - 1` and `to_slice`, with the `+ 1` and the `- 1` already cancelled.
    let from_slice = lowerbound_elm / lanes_per_slice;
    let to_slice = upperbound_elm / lanes_per_slice;

    // ⛔ THE UPPER END FIRST, because it is the one that shifts: see this function's third stop.
    if to_slice < 0 || to_slice >= slices_per_stick {
        return None;
    }

    // ⛔ AND A NEGATIVE START IS THE UNSIGNED WRAP, whose observable result is an empty loop.
    if from_slice < 0 {
        return Some(SliceMask(0));
    }

    // `for (unsigned i = from_slice - 1; i <= to_slice; i++) mask_val += 1 << i;` — a range that runs
    // backwards is empty, which is the all-lanes-live answer.
    let mut mask_val = 0;
    for i in u32::try_from(from_slice).ok()?..=u32::try_from(to_slice).ok()? {
        mask_val += 1 << i;
    }

    Some(SliceMask(mask_val))
}

// ═══════════════════════════════════════════ 164/384 ═══════════════════════════════════════════

/// HOW MANY TIMES A PACK OR SHUFFLE PATTERN REPEATS — `is_any_of(repetition, 4, 8)`
/// (`VectorChainHelper.cpp:142-146`).
///
/// ⛔⛔ TWO VALUES, AND THE OP'S FIELD IS A `u32`. `vectorchain.pack`'s `repetition` is an
/// `IndexAttr` and `vectorchain.shuffle`'s an `i32` — the island keeps both as they are printed — so
/// the closed set lives here, at the one place that decides it, and the refusal *"the cases with
/// !is_any_of(repetition, 4, 8) are not supported yet"* is spent once.
///
/// ⭐ AND `Four` MATCHES NO ROW OF THE MERGE-AND-PACK TABLE. Every one of the thirty-four rows takes
/// the member default `repetition = 8` and [`merge_type_from_indices`] compares for equality
/// (`:399`), so a repetition of four reaches that function and finds nothing — it is a GCVT/FCVT
/// shuffle's number (entry 277), which is why this gate accepts it and the table does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PackRepetition {
    /// `repetition = 4`.
    Four,
    /// `repetition = 8` — the whole merge-and-pack table's value.
    Eight,
}

impl PackRepetition {
    /// `is_any_of(repetition, 4, 8)`, as a witness.
    #[must_use]
    pub const fn of(repetition: u32) -> Option<PackRepetition> {
        match repetition {
            4 => Some(PackRepetition::Four),
            8 => Some(PackRepetition::Eight),
            _ => None,
        }
    }

    /// The repetition itself.
    #[must_use]
    pub const fn count(self) -> u32 {
        match self {
            PackRepetition::Four => 4,
            PackRepetition::Eight => 8,
        }
    }
}

/// EITHER OF THE TWO OPS THIS CHECK ACCEPTS, with the three things it reads off them.
///
/// ⛔⛔ THIS IS `DT_CHECK_MSG((pack_op || shuffle_op), "Expect a pack or shuffle op.")`
/// (`VectorChainHelper.cpp:150`) DISCHARGED BY CONSTRUCTION. The reference takes a bare
/// `Operation*`, casts to both, aborts if neither, and then writes the SAME THREE READS TWICE with a
/// ternary — `pack_op ? pack_op.getIndices() : shuffle_op.getIndices()`, and again for the type
/// (`:151-153`). A witness minted from the op does the discrimination once, and the reference's abort
/// becomes the `None` at [`PackOrShuffle::of`], the way [`super::vc_helper::PtUnit::of`] handles its
/// own component check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackOrShuffle<'a> {
    /// `getIndices()` — one entry per element of the pattern.
    indices: &'a [i32],
    /// `getRepetition()`, still as the op prints it.
    repetition: u32,
    /// `getType()` — the RESULT type, which is what `getTotalVectorSize` is asked about.
    ty: Vector,
}

impl<'a> PackOrShuffle<'a> {
    /// A PACK OR A SHUFFLE, or `None` for anything else — the reference's `DT_CHECK_MSG`.
    ///
    /// ⭐ THE REPETITION COMES FROM THE OP, and that is not a shortcut: all four call sites pass the
    /// op's own — `int repetition = shuffle_op.getRepetition();`
    /// (`VectorChainToSentientPESFP.cpp:634-638`), `pack_op.getRepetition().getSExtValue()`
    /// (`:688-692`), and the same two in `VectorChainToSentientPT.cpp:629`, `:822`. Passing it
    /// separately would let a caller validate one op's indices against another's repetition.
    #[must_use]
    pub fn of(op: &'a vc::Op) -> Option<PackOrShuffle<'a>> {
        match op {
            vc::Op::Pack {
                indices,
                repetition,
                ty,
                ..
            }
            | vc::Op::Shuffle {
                indices,
                repetition,
                ty,
                ..
            } => Some(PackOrShuffle {
                indices,
                repetition: *repetition,
                ty: *ty,
            }),
            _ => None,
        }
    }

    /// The result type the indices have to fill.
    #[must_use]
    pub const fn ty(&self) -> Vector {
        self.ty
    }
}

/// AN INDEX LIST THAT HAS PASSED [`check_validity_of_pack_and_shuffle_lowering`], with the
/// repetition it was checked against.
///
/// ⛔⛔ IT IS THE REFERENCE'S OUT-PARAMETER, AND A TYPE IS WHAT MAKES THE ORDER UNSKIPPABLE. The C++
/// signature is `checkValidityOfPackAndShuffleLowering(Operation* op, std::vector<int>& indices, int
/// repetition)`: the caller declares an EMPTY vector, the callee fills it, and the caller then hands
/// that same vector to `getMergeTypeFromIndices` or
/// `getGCVTorFCVTTypeFromIndicesAndCastInputs` (`VectorChainToSentientPESFP.cpp:687-708`). Nothing
/// but call order stops those two from being handed an unchecked list. Here they take this, and only
/// this function makes one.
///
/// ⭐ IT BORROWS THE OP'S OWN LIST RATHER THAN COPYING IT, and that is exact: the loop at `:155-167`
/// pushes EVERY element and fails on the first one it cannot accept, so on success the filled vector
/// is elementwise the op's `indices` attribute. ⛔ The `else` arm of that loop — *"Indices should be
/// integers"* — is unrepresentable here, because [`vc::Op::Pack::indices`] is a `Vec<i32>` and an
/// attribute of another kind could not have been parsed into it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValidPackIndices<'a> {
    /// The op's own indices, all `>= -1`.
    indices: &'a [i32],
    /// The repetition they were checked against.
    repetition: PackRepetition,
}

impl<'a> ValidPackIndices<'a> {
    /// The checked indices, where `-1` is a lane the instruction does not write.
    #[must_use]
    pub const fn indices(&self) -> &'a [i32] {
        self.indices
    }

    /// The repetition they were checked against.
    #[must_use]
    pub const fn repetition(&self) -> PackRepetition {
        self.repetition
    }
}

/// Replaces: e164_checkValidityOfPackAndShuffleLowering
///
/// # WHETHER A PACK OR SHUFFLE CAN BE LOWERED AT ALL — the four gates ahead of naming an instruction
///
/// ```cpp
/// mlir::LogicalResult vectorchain::checkValidityOfPackAndShuffleLowering(
///     Operation* op, std::vector<int>& indices, int repetition) {
///   if (!is_any_of(repetition, 4, 8)) {
///     op->emitError("The cases with !is_any_of(repetition, 4, 8) are not supported yet");
///     return failure();
///   }
///   auto pack_op = llvm::dyn_cast<vectorchain::PackOp>(op);
///   auto shuffle_op = llvm::dyn_cast<vectorchain::ShuffleOp>(op);
///   DT_CHECK_MSG((pack_op || shuffle_op), "Expect a pack or shuffle op.");
///   auto op_indices = pack_op ? pack_op.getIndices().getValue()
///                             : shuffle_op.getIndices().getValue();
///   Type type = pack_op ? pack_op.getType() : shuffle_op.getType();
///   int vec_size = dcc::utils::getTotalVectorSize(type);
///   for (auto index_iter : op_indices) {
///     if (auto index_attr = mlir::dyn_cast<IntegerAttr>(index_iter)) {
///       int index = index_attr.getInt();
///       if (index < -1) {
///         op->emitError("Indices should be integers >= -1");
///         return failure();
///       }
///       indices.push_back(index);
///     } else {
///       op->emitError("Indices should be integers");
///       return failure();
///     }
///   }
///   if (repetition * indices.size() != vec_size) {
///     op->emitError("Size of array indices multiply to repetition should be equal to "
///                   "the size of vector");
///     return failure();
///   }
///   for (auto index : indices) {
///     if (index > 2 * (int)indices.size()) {
///       op->emitError("None of the elements of the indices array can be greater than 2 "
///                     "times the size of the array");
///       op->dump();
///       return failure();
///     }
///   }
///   return success();
/// }
/// ```
/// (`dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:140-185`)
///
/// # ⭐ THE PATTERN TILES THE RESULT — `repetition * indices.size() == vec_size`
///
/// A `vectorchain.pack` states one period of its permutation and a repetition count, and the two
/// have to fill the result exactly: 16 indices × 8 for a `vector<128xi8>`, 8 × 8 for a
/// `vector<64xi16>`, 32 × 8 for a `vector<256xi4>`, 64 × 8 for a `vector<512xi2>` — all four widths
/// of `dcc/test/SFP/merge_and_pack.mlir:161-266`. ⛔ AND `vec_size` IS THE **RESULT** TYPE'S
/// element count (`:153-154`), not an operand's; on those fixtures all three agree, and where a pack
/// narrows they would not.
///
/// # ⛔ THE LAST GATE IS **STRICTLY** GREATER, AND IT IS LOOSER THAN IT LOOKS
///
/// `index > 2 * (int)indices.size()` (`:177`) admits `index == 2 * size` — for eight indices, 16 —
/// even though the two operands concatenated hold exactly `2 * size` elements, whose last index is
/// `2 * size - 1`. So one out-of-range index per shape passes this check. ⭐ REPRODUCED, NOT
/// TIGHTENED: nothing downstream indexes a buffer with these — [`merge_type_from_indices`] compares
/// them against a table and answers `None` for a list no instruction implements — so the loose bound
/// costs a diagnostic, not a read out of bounds, and closing it here would be this port inventing a
/// rule. `pack1`'s `15` is the largest any fixture uses.
///
/// ⛔ AND THE ORDER OF THE TWO LOOPS MATTERS. The `>= -1` test is per element DURING the copy and the
/// `2 * size` test is a second pass over the whole list (`:155-167` then `:176-184`), so a list that
/// is both too short and holds a `-2` is refused for the `-2` — which is the diagnostic a reader gets
/// and therefore part of the answer.
#[must_use]
pub fn check_validity_of_pack_and_shuffle_lowering(
    op: PackOrShuffle<'_>,
) -> Option<ValidPackIndices<'_>> {
    // `if (!is_any_of(repetition, 4, 8))` — "The cases with !is_any_of(repetition, 4, 8) are not
    // supported yet".
    let repetition = PackRepetition::of(op.repetition)?;

    // `int vec_size = dcc::utils::getTotalVectorSize(type);` — ⭐ THE PRODUCT OF THE SHAPE
    // (`dcc/src/Utils/Utils.cpp:704-714`), which for this island's one-dimensional [`Vector`] is
    // its length.
    let vec_size = op.ty.len;

    // `for (auto index_iter : op_indices) { … if (index < -1) … indices.push_back(index); }`
    for &index in op.indices {
        // `op->emitError("Indices should be integers >= -1")`
        if index < -1 {
            return None;
        }
    }

    // `if (repetition * indices.size() != vec_size)`
    if u64::from(repetition.count()) * op.indices.len() as u64 != vec_size {
        return None;
    }

    // `for (auto index : indices) if (index > 2 * (int)indices.size())` — strictly greater; see the
    // note above.
    let bound = 2 * i64::try_from(op.indices.len()).ok()?;
    for &index in op.indices {
        if i64::from(index) > bound {
            return None;
        }
    }

    // `return success();` — with the checked list, so its two consumers cannot be handed another.
    Some(ValidPackIndices {
        indices: op.indices,
        repetition,
    })
}

// ═══════════════════════════════════════════ 165/384 ═══════════════════════════════════════════

/// THE THIRTY-FOUR MERGE-AND-PACK INSTRUCTIONS THE SFP IMPLEMENTS, IN THE ORDER
/// `getMergeTypeFromIndices` SCANS THEM (`VectorChainHelper.cpp:328-389`).
///
/// # ⛔⛔ THE ORDER IS LOAD-BEARING, AND IBM'S OWN TEST PROVES IT
///
/// `pack0` and `pack16` have IDENTICAL rows — `{0, 1, 2, 3, 4, 5, 6, 7}` at 16 bits, same repetition,
/// same `sign_extend` (`:374`, `:382`) — and the scan returns the FIRST match, so `pack16` is
/// unreachable. `dcc/test/SFP/merge_and_pack.mlir:232` writes a pack it names `%pack16` and the
/// reference lowers it to `binary_operator pack0` (the 24th `binary_operator` of that file's
/// `CHECK-SENT-IR`). Sorting this table, or making it a map, would change that answer.
///
/// ⭐ SO IT IS A `Vec` AND NOT A `BTreeMap`, and the sole test of a row's identity is its position.
///
/// ⭐ WHY IT IS BUILT PER CALL, as the C++ does (a function-local `std::vector` initialised on every
/// entry, `:328`): the rows own their index vectors, `MergeAndPack` is not `Copy`, and this is called
/// once per pack op in a program. A `LazyLock` would be the same table with a `static` in front of
/// it; nothing here mutates a row, so there is nothing to protect.
#[must_use]
pub fn merge_and_pack_insts() -> Vec<MergeAndPack> {
    vec![
        // ── 2-BIT ELEMENTS — sixteen real lanes and forty-eight pads (`:329-333`).
        MergeAndPack::new(
            MergeOrPack::Pack(sen::PackIndex::P24),
            Bits(2),
            vec![
                0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1,
            ],
            false,
        ),
        // ── 4-BIT ELEMENTS (`:334-339`).
        MergeAndPack::new(
            MergeOrPack::Pack(sen::PackIndex::P8),
            Bits(4),
            vec![
                0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            ],
            false,
        ),
        MergeAndPack::new(
            MergeOrPack::Pack(sen::PackIndex::P9),
            Bits(4),
            vec![
                0, 32, -1, -1, 4, 36, -1, -1, 8, 40, -1, -1, 12, 44, -1, -1, 16, 48, -1, -1, 20,
                52, -1, -1, 24, 56, -1, -1, 28, 60, -1, -1,
            ],
            false,
        ),
        // ── 8-BIT ELEMENTS (`:340-366`).
        MergeAndPack::new(
            MergeOrPack::Merge {
                width: sen::MergeWidth::W8,
                high: true,
            },
            Bits(8),
            vec![1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31],
            false,
        ),
        MergeAndPack::new(
            MergeOrPack::Merge {
                width: sen::MergeWidth::W8,
                high: false,
            },
            Bits(8),
            vec![0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30],
            false,
        ),
        MergeAndPack::new(
            MergeOrPack::Pack(sen::PackIndex::P12),
            Bits(8),
            vec![0, -1, 1, -1, 2, -1, 3, -1, 4, -1, 5, -1, 6, -1, 7, -1],
            false,
        ),
        MergeAndPack::new(
            MergeOrPack::Pack(sen::PackIndex::P13),
            Bits(8),
            vec![8, -1, 9, -1, 10, -1, 11, -1, 12, -1, 13, -1, 14, -1, 15, -1],
            false,
        ),
        // ⛔ `pack14` AND `pack15` ARE `pack12` AND `pack13` WITH `sign_extend` — the only two rows
        // of the thirty-four that set it (`:350-357`), and the only reason the scan needs the
        // `inst.sign_extend == sign_extend` test at all.
        MergeAndPack::new(
            MergeOrPack::Pack(sen::PackIndex::P14),
            Bits(8),
            vec![0, -1, 1, -1, 2, -1, 3, -1, 4, -1, 5, -1, 6, -1, 7, -1],
            true,
        ),
        MergeAndPack::new(
            MergeOrPack::Pack(sen::PackIndex::P15),
            Bits(8),
            vec![8, -1, 9, -1, 10, -1, 11, -1, 12, -1, 13, -1, 14, -1, 15, -1],
            true,
        ),
        MergeAndPack::new(
            MergeOrPack::Pack(sen::PackIndex::P25),
            Bits(8),
            vec![0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
            false,
        ),
        MergeAndPack::new(
            MergeOrPack::Pack(sen::PackIndex::P26),
            Bits(8),
            vec![0, 2, 4, 6, 16, 18, 20, 22, 8, 10, 12, 14, 24, 26, 28, 30],
            false,
        ),
        MergeAndPack::new(
            MergeOrPack::Pack(sen::PackIndex::P27),
            Bits(8),
            vec![0, 2, 16, 18, 4, 6, 20, 22, 8, 10, 24, 26, 12, 14, 28, 30],
            false,
        ),
        // ── 16-BIT ELEMENTS (`:368-389`).
        MergeAndPack::new(
            MergeOrPack::Merge {
                width: sen::MergeWidth::W16,
                high: true,
            },
            Bits(16),
            vec![1, 9, 3, 11, 5, 13, 7, 15],
            false,
        ),
        MergeAndPack::new(
            MergeOrPack::Merge {
                width: sen::MergeWidth::W16,
                high: false,
            },
            Bits(16),
            vec![0, 8, 2, 10, 4, 12, 6, 14],
            false,
        ),
        MergeAndPack::new(
            MergeOrPack::Merge {
                width: sen::MergeWidth::W32,
                high: true,
            },
            Bits(16),
            vec![2, 3, 10, 11, 6, 7, 14, 15],
            false,
        ),
        MergeAndPack::new(
            MergeOrPack::Merge {
                width: sen::MergeWidth::W32,
                high: false,
            },
            Bits(16),
            vec![0, 1, 8, 9, 4, 5, 12, 13],
            false,
        ),
        MergeAndPack::new(
            MergeOrPack::Merge {
                width: sen::MergeWidth::W64,
                high: true,
            },
            Bits(16),
            vec![4, 5, 6, 7, 12, 13, 14, 15],
            false,
        ),
        MergeAndPack::new(
            MergeOrPack::Merge {
                width: sen::MergeWidth::W64,
                high: false,
            },
            Bits(16),
            vec![0, 1, 2, 3, 8, 9, 10, 11],
            false,
        ),
        MergeAndPack::new(
            MergeOrPack::Pack(sen::PackIndex::P0),
            Bits(16),
            vec![0, 1, 2, 3, 4, 5, 6, 7],
            false,
        ),
        MergeAndPack::new(
            MergeOrPack::Pack(sen::PackIndex::P1),
            Bits(16),
            vec![15, 1, 2, 3, 4, 5, 6, 7],
            false,
        ),
        MergeAndPack::new(
            MergeOrPack::Pack(sen::PackIndex::P2),
            Bits(16),
            vec![14, 15, 2, 3, 4, 5, 6, 7],
            false,
        ),
        MergeAndPack::new(
            MergeOrPack::Pack(sen::PackIndex::P3),
            Bits(16),
            vec![13, 14, 15, 3, 4, 5, 6, 7],
            false,
        ),
        MergeAndPack::new(
            MergeOrPack::Pack(sen::PackIndex::P4),
            Bits(16),
            vec![12, 13, 14, 15, 4, 5, 6, 7],
            false,
        ),
        MergeAndPack::new(
            MergeOrPack::Pack(sen::PackIndex::P5),
            Bits(16),
            vec![11, 12, 13, 14, 15, 5, 6, 7],
            false,
        ),
        MergeAndPack::new(
            MergeOrPack::Pack(sen::PackIndex::P6),
            Bits(16),
            vec![10, 11, 12, 13, 14, 15, 6, 7],
            false,
        ),
        MergeAndPack::new(
            MergeOrPack::Pack(sen::PackIndex::P7),
            Bits(16),
            vec![9, 10, 11, 12, 13, 14, 15, 7],
            false,
        ),
        // ⛔ THE UNREACHABLE ROW — `pack0`'s twin, kept in place because removing it would change
        // nothing and pretending it is absent would misdescribe the table. See this function's note.
        MergeAndPack::new(
            MergeOrPack::Pack(sen::PackIndex::P16),
            Bits(16),
            vec![0, 1, 2, 3, 4, 5, 6, 7],
            false,
        ),
        MergeAndPack::new(
            MergeOrPack::Pack(sen::PackIndex::P17),
            Bits(16),
            vec![8, 0, 1, 2, 3, 4, 5, 6],
            false,
        ),
        MergeAndPack::new(
            MergeOrPack::Pack(sen::PackIndex::P18),
            Bits(16),
            vec![8, 9, 0, 1, 2, 3, 4, 5],
            false,
        ),
        MergeAndPack::new(
            MergeOrPack::Pack(sen::PackIndex::P19),
            Bits(16),
            vec![8, 9, 10, 0, 1, 2, 3, 4],
            false,
        ),
        MergeAndPack::new(
            MergeOrPack::Pack(sen::PackIndex::P20),
            Bits(16),
            vec![8, 9, 10, 11, 0, 1, 2, 3],
            false,
        ),
        MergeAndPack::new(
            MergeOrPack::Pack(sen::PackIndex::P21),
            Bits(16),
            vec![8, 9, 10, 11, 12, 0, 1, 2],
            false,
        ),
        MergeAndPack::new(
            MergeOrPack::Pack(sen::PackIndex::P22),
            Bits(16),
            vec![8, 9, 10, 11, 12, 13, 0, 1],
            false,
        ),
        MergeAndPack::new(
            MergeOrPack::Pack(sen::PackIndex::P23),
            Bits(16),
            vec![8, 9, 10, 11, 12, 13, 14, 0],
            false,
        ),
    ]
}

/// Replaces: e165_getMergeTypeFromIndices
///
/// # WHICH MERGE OR PACK INSTRUCTION A PERMUTATION IS — the table scan, and `""` for *"there is no
/// merge or pack instruction corresponding to the following operation"*
///
/// ```cpp
/// std::string vectorchain::getMergeTypeFromIndices(std::vector<int> indices,
///                                                  int repetition,
///                                                  bool sign_extend,
///                                                  unsigned element_bit_width) {
///   struct merge_and_pack_type { … };                 // ported at `MergeAndPack::new` (entry 064)
///   std::vector<merge_and_pack_type> merge_and_pack_insts = { … };  // `merge_and_pack_insts` above
///
///   int indices_sum = 0;
///   for (auto index : indices) indices_sum += index;
///
///   for (auto merge_and_pack_inst : merge_and_pack_insts) {
///     if (element_bit_width > merge_and_pack_inst.element_bit_width) continue;
///     int scale = merge_and_pack_inst.element_bit_width / element_bit_width;
///     bool found = true;
///     if (indices.size() != merge_and_pack_inst.size * scale) continue;
///     if (repetition != merge_and_pack_inst.repetition) continue;
///     if (scale == 1 && indices_sum != merge_and_pack_inst.sum) continue;
///
///     for (int i = 0; i < merge_and_pack_inst.size && found; i++) {
///       for (int j = 0; j < scale && found; j++) {
///         if (indices[i * scale + j] != merge_and_pack_inst.vec[i] * scale + j)
///           found = false;
///       }
///     }
///
///     if (found && merge_and_pack_inst.sign_extend == sign_extend)
///       return merge_and_pack_inst.name;
///   }
///   return "";
/// }
/// ```
/// (`dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:301-412` — the
/// thirty-four-row initialiser at `:328-389` is [`merge_and_pack_insts`] and the local struct at
/// `:305-326` is [`MergeAndPack`], so only the scan is repeated here)
///
/// # ⭐⭐ `scale` IS WHY ONE TABLE SERVES FOUR ELEMENT WIDTHS
///
/// Every row is written at the widest element it applies to, and a NARROWER element reaches the same
/// instruction by moving `scale` neighbours at a time: at `element_bit_width == 8` a 16-bit row's
/// index `k` becomes the pair `2k, 2k+1`, so `merge32l`'s `{0, 1, 8, 9, …}` also names the 8-bit
/// permutation `{0, 1, 2, 3, 16, 17, 18, 19, …}`. That is exactly what the inner double loop tests —
/// `indices[i * scale + j] == vec[i] * scale + j` for every `j` — and it is why the row's `size` is
/// multiplied by `scale` before the lengths are compared.
///
/// ⛔ A ROW WITH A `-1` IN IT CAN THEREFORE ONLY MATCH AT `scale == 1`. `pack12`'s pad would demand
/// `-1 * 2 + 0 == -2` at 4-bit elements, and [`check_validity_of_pack_and_shuffle_lowering`] refuses
/// any index below `-1` before this is ever called — so the pads are not "expanded", they simply
/// stop the widening. Nothing in the reference says so; the arithmetic does.
///
/// ⛔ AND THE CHECKSUM FILTER IS GATED ON `scale == 1` FOR THE SAME REASON. `indices_sum` compares
/// against a sum taken at the row's own width; at `scale > 1` the list is longer and its sum is
/// larger, so the filter is skipped rather than scaled (`:400`) — a fast path only, since the
/// per-lane loop decides.
///
/// # ⛔ `element_bit_width` IS THE **RESULT** ELEMENT'S WIDTH, AND ITS ONLY CALLER SAYS SO
///
/// `getMergeTypeFromIndices(indices, repetition, pack_op.getSignExtend(),
/// dataflow::utils::getElementTypeBitWidth(pack_op.getResult().getType()))`
/// (`VectorChainToSentientPESFP.cpp:706-708`) — reached only when NEITHER operand comes from a
/// `vectorchain.cast` (`:694-702`); with casts it is `getGCVTorFCVTTypeFromIndicesAndCastInputs`
/// (entry 277) instead. ⭐ AND THAT CALL SITE ALSO SETS `op_info.compute_precision_ = "none"`,
/// because *"MERGE/PACK instructions are bitwise operations"* (`:703-705`) — the precision belongs to
/// the caller's own port and not to this answer.
///
/// # THE ONE STOP THIS PORT ADDS
///
/// ⛔ `element_bit_width == 0` DIVIDES BY ZERO AT `:396`. `unsigned` there, and nothing on the path
/// excludes it: `getElementTypeBitWidth` of a zero-width element would reach the first row with
/// `0 > 2` false and then divide. A vector of zero-bit elements is not a pack this lowering can name,
/// so it answers `None` — the reference would trap.
#[must_use]
pub fn merge_type_from_indices(
    indices: &ValidPackIndices<'_>,
    sign_extend: bool,
    element_bit_width: Bits,
) -> Option<MergeOrPack> {
    // ⛔ THE DIVISOR — see this function's last note.
    if element_bit_width.0 == 0 {
        return None;
    }

    let repetition = indices.repetition().count();
    let indices = indices.indices();

    // `int indices_sum = 0; for (auto index : indices) indices_sum += index;`
    let indices_sum: i32 = indices.iter().sum();

    for inst in merge_and_pack_insts() {
        // `if (element_bit_width > merge_and_pack_inst.element_bit_width) continue;` — a row cannot
        // be narrowed, only widened.
        if element_bit_width.0 > inst.element_bit_width().0 {
            continue;
        }

        // `int scale = merge_and_pack_inst.element_bit_width / element_bit_width;` — ⛔ TRUNCATING,
        // and the reference's own division: an element width that does not divide the row's (a
        // 3-bit element against a 16-bit row) gives 5, and the length test below then rejects it.
        let scale = (inst.element_bit_width().0 / element_bit_width.0) as usize;

        // `if (indices.size() != merge_and_pack_inst.size * scale) continue;`
        if indices.len() != inst.size() * scale {
            continue;
        }

        // `if (repetition != merge_and_pack_inst.repetition) continue;`
        if repetition != inst.repetition() {
            continue;
        }

        // `if (scale == 1 && indices_sum != merge_and_pack_inst.sum) continue;`
        if scale == 1 && indices_sum != inst.sum() {
            continue;
        }

        // The per-lane comparison — `found` starts true and both loops stop at the first mismatch.
        // ⭐ THE CASTS ARE EXACT: `scale` is at most sixteen — the widest row divided by a
        // one-bit element — and `j` is below it.
        let scale_step = scale as i32;
        let mut found = true;
        for i in 0..inst.size() {
            for j in 0..scale {
                if indices[i * scale + j] != inst.vec()[i] * scale_step + j as i32 {
                    found = false;
                    break;
                }
            }
            if !found {
                break;
            }
        }

        // `if (found && merge_and_pack_inst.sign_extend == sign_extend) return inst.name_;`
        if found && inst.sign_extend() == sign_extend {
            return Some(inst.name());
        }
    }

    // `return "";` — no instruction implements this permutation.
    None
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 228/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e228_validateLoweringAndSetMissingParameters
///
/// **228/384** `vectorchain::validateLoweringAndSetMissingParameters` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:483` (49L).
///
/// The pre-order walk that closes the vectorchain lowering: nine op kinds must be GONE by now, and
/// every compute operand still carrying `-1` gets the next data id, counting up from
/// [`OperandReuse::total_data_origins_count`].
///
/// ⛔ THE COUNTER IS SHARED ACROSS THE WHOLE UNIT, not per op — A, B then C of one mac, then the next
/// compute's — so a walk in another order renumbers every operand the reuse table already knows.
pub fn validate_lowering_and_set_missing_parameters<A: Arch>(
    unit: &mut SentientProgramUnit<A>,
    reuse_info: &OperandReuse,
) {
    let mut total_reuse_ids = reuse_info.total_data_origins_count();
    set_missing_data_ids(&mut unit.body, &mut total_reuse_ids);
}

/// The `unit.walk<WalkOrder::PreOrder>` of [`validate_lowering_and_set_missing_parameters`], over one
/// block.
fn set_missing_data_ids(ops: &mut [SenOp], total_reuse_ids: &mut u32) {
    /// `if (mac_op.getOpADataID() == -1) { .. total_reuse_ids++; }` — ⛔ `None` IS that `-1`; a
    /// [`DataId`] carries the sentinel so this line does not have to name it.
    fn assign(operand: &mut sen::Operand, total_reuse_ids: &mut u32) {
        if operand.data_id.is_none() {
            operand.data_id = Some(DataId::Assigned(DataOriginId(*total_reuse_ids)).attribute());
            *total_reuse_ids += 1;
        }
    }

    for op in ops {
        match op {
            // ⛔⛔ THE NINE KINDS THE REFERENCE REFUSES, as a build failure rather than an
            // `emitError` + `WalkResult::interrupt()`. Each names the op, and none may be stood in
            // for: an unlowered `vectorchain.multiply_and_accumulate` left in a unit is a program
            // dbo-opt accepts and mislowers.
            SenOp::Dataflow(
                dfir_op::dataflow::Op::Send { .. } | dfir_op::dataflow::Op::Receive { .. },
            ) => todo!(
                "dataflow.send/dataflow.receive: unable to lower the op into Sentient as part of \
                 the vectorchain lowering"
            ),
            SenOp::Vector(dfir_op::vector::Op::Load { .. } | dfir_op::vector::Op::Store { .. }) => {
                todo!(
                    "vector.load/vector.store: unable to lower the op into Sentient as part of the \
                     vectorchain lowering"
                )
            }
            SenOp::Agen(
                dfir_op::agen::Op::VectorLoad { .. } | dfir_op::agen::Op::VectorStore { .. },
            ) => todo!(
                "agen.vector_load/agen.vector_store: unable to lower the op into Sentient as part \
                 of the vectorchain lowering"
            ),
            SenOp::VectorChain(
                vc::Op::MultiplyAccumulate { .. } | vc::Op::Multiply { .. } | vc::Op::Binary { .. },
            ) => todo!(
                "vectorchain.multiply_and_accumulate/multiply/binary: unable to lower the op into \
                 Sentient as part of the vectorchain lowering"
            ),

            // ⭐ A, B, THEN C, IN THAT ORDER — the reference's three `if`s, and the order is what the
            // ids mean.
            SenOp::Sentient(sen::Op::VectorMac {
                op_a, op_b, op_c, ..
            }) => {
                assign(op_a, total_reuse_ids);
                assign(op_b, total_reuse_ids);
                assign(op_c, total_reuse_ids);
            }
            // `else if (auto bin_op = llvm::dyn_cast<sentient::BinaryOp>(op))` — A then B.
            SenOp::Sentient(sen::Op::VectorBinary { op_a, op_b, .. }) => {
                assign(op_a, total_reuse_ids);
                assign(op_b, total_reuse_ids);
            }
            SenOp::Sentient(sen::Op::For { body, .. }) => {
                set_missing_data_ids(body, total_reuse_ids);
            }
            SenOp::Sentient(sen::Op::If {
                then_body,
                else_body,
                ..
            }) => {
                set_missing_data_ids(then_body, total_reuse_ids);
                set_missing_data_ids(else_body, total_reuse_ids);
            }
            // `return WalkResult::advance();` — every other op, including the ternary and unary
            // computes the reference's two `dyn_cast`s do not name.
            _ => {}
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 229/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// A NON-PT MASK CONSTANT AND THE VALUE IT BINDS — what [`get_mask_value_for_non_pt`] hands back.
///
/// ⛔ TWO FIELDS BECAUSE THE `Value` ALONE CANNOT BE PLACED. The reference returns the op AS a
/// `Value`; here the caller must put [`Self::op`] where `builder.setInsertionPoint(op)` put it and use
/// [`Self::value`] as the compute's mask operand. Same shape as
/// [`super::vc_helper::MaskValue::Constant`], the PT side of the same question.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NonPtMask {
    /// The `sentient.scalar_constant` to insert immediately before the masked compute.
    pub op: SenOp,
    /// The value it binds.
    pub value: Val,
}

/// Replaces: e229_getMaskValueForNonPT
///
/// **229/384** `vectorchain::getMaskValueForNonPT` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:114` (9L).
///
/// [`get_mask_value_constant_for_non_pt`]'s answer as a `sentient.constant` of type `index`, or
/// nothing when there is no constant to be had.
///
/// ⛔ `builder.setInsertionPoint(op)` IS THE DROPPED MECHANISM, and placing is the only use the
/// reference has for `op` besides its location — [`NonPtMask::op`] is the obligation it leaves the
/// caller. The builder itself is NOT dropped: `getIndexType()` is [`ScalarTy::Index`] below.
#[must_use]
pub fn get_mask_value_for_non_pt<A: Arch>(mask: &vc::Op, values: &mut Values) -> Option<NonPtMask> {
    let mask_val = get_mask_value_constant_for_non_pt::<A>(mask)?;
    let value = values.mint();
    Some(NonPtMask {
        op: SenOp::Sentient(sen::Op::ScalarConstant {
            value: i64::from(mask_val.bits()),
            result: value,
            // ⭐ `builder.getIndexType()`, and the immediate locale every mask constant takes — the
            // PT side mints exactly this op (`vc_helper::get_mask_value_for_pt`).
            reg_locale: sen::RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        }),
        value,
    })
}

/// THE ELEVEN SPELLINGS THE CVT TABLES ARE WRITTEN IN — `convertStringToType`'s accepted strings
/// (`VectorChainHelper.hpp:196-219`).
///
/// ⛔ NOT MLIR'S TYPE SPELLINGS, AND THE DIFFERENCE IS WHY THIS ENUM EXISTS RATHER THAN
/// [`ElemType::spelling`]: the float widths are `fp16`/`fp32`, not `f16`/`f32`, while the fp8 and
/// integer names ARE the MLIR ones. `getGCVTorFCVTTypeFromIndicesAndCastInputs`' table is written in
/// these (`"fp32"`, `"f8E5M2"`, `"bf16"` at `VectorChainHelper.cpp:245-267`), and this crate's rule
/// is that a closed set is an enum rather than a string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeSpelling {
    /// `fp16` — `f16`.
    Fp16,
    /// `bf16`.
    Bf16,
    /// `fp32` — `f32`.
    Fp32,
    /// `f8E4M3FN`.
    F8E4M3Fn,
    /// `f8E5M2`.
    F8E5M2,
    /// `mxfp4`.
    Mxfp4,
    /// `mxfp8`.
    Mxfp8,
    /// `mxint4` — ⛔ ACCEPTED BY ENTRY 230 AND UNPRODUCIBLE BY ENTRY 231. See
    /// [`convert_type_to_string`].
    Mxint4,
    /// `i16`.
    I16,
    /// `i8`.
    I8,
    /// `i4`.
    I4,
}

/// Replaces: e230_convertStringToType
///
/// **230/384** `convertStringToType` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:196` (24L).
///
/// The eleven-armed `if`/`else if` chain, as a total function: `llvm_unreachable("unknown type
/// string")` (`:220`) is unrepresentable once the argument is [`TypeSpelling`] rather than a
/// `std::string`. ⚠️ DEAD IN THE REFERENCE — nothing calls it; its inverse
/// [`convert_type_to_string`] has the one caller (entry 277).
#[must_use]
pub const fn convert_string_to_type(type_string: TypeSpelling) -> ElemType {
    match type_string {
        TypeSpelling::Fp16 => ElemType::F16,
        TypeSpelling::Bf16 => ElemType::Bf16,
        TypeSpelling::Fp32 => ElemType::F32,
        TypeSpelling::F8E4M3Fn => ElemType::F8E4M3Fn,
        TypeSpelling::F8E5M2 => ElemType::F8E5M2,
        // `dataflow::CustomMXFloatType::get(ctx, 4)` / `(ctx, 8)`.
        TypeSpelling::Mxfp4 => ElemType::MxFloat(4),
        TypeSpelling::Mxfp8 => ElemType::MxFloat(8),
        // `dataflow::CustomMXIntType::get(ctx, 4)` — a DIFFERENT type from `mxfp4`, same width.
        TypeSpelling::Mxint4 => ElemType::MxInt(4),
        TypeSpelling::I16 => ElemType::Int(16),
        TypeSpelling::I8 => ElemType::Int(8),
        TypeSpelling::I4 => ElemType::Int(4),
    }
}

/// Replaces: e231_convertTypeToString
///
/// **231/384** `convertTypeToString` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:223` (22L).
///
/// ⛔⛔ TEN ARMS AGAINST ENTRY 230'S ELEVEN — `mxint4` IS MISSING (`:223-244`), so a type
/// [`convert_string_to_type`] produces reaches `llvm_unreachable("unknown type")` on the way back.
/// ⛔ `None` IS THAT ABORT, and it composes: the one caller compares the answer with a table string
/// and clears `found` on a mismatch (`VectorChainHelper.cpp:285-292`), which is what `None` does.
#[must_use]
pub const fn convert_type_to_string(ty: ElemType) -> Option<TypeSpelling> {
    match ty {
        ElemType::F16 => Some(TypeSpelling::Fp16),
        ElemType::Bf16 => Some(TypeSpelling::Bf16),
        ElemType::F32 => Some(TypeSpelling::Fp32),
        ElemType::F8E4M3Fn => Some(TypeSpelling::F8E4M3Fn),
        ElemType::F8E5M2 => Some(TypeSpelling::F8E5M2),
        ElemType::MxFloat(4) => Some(TypeSpelling::Mxfp4),
        ElemType::MxFloat(8) => Some(TypeSpelling::Mxfp8),
        ElemType::Int(16) => Some(TypeSpelling::I16),
        ElemType::Int(8) => Some(TypeSpelling::I8),
        ElemType::Int(4) => Some(TypeSpelling::I4),
        // `llvm_unreachable("unknown type")` — ⛔ `MxInt` IS IN HERE, and `f8E8M0FNU`, `f4E2M1FN`
        // and every other integer width with it.
        ElemType::Int(_)
        | ElemType::F8E8M0Fnu
        | ElemType::F4E2M1Fn
        | ElemType::MxFloat(_)
        | ElemType::MxInt(_) => None,
    }
}

// ═══════════════════════════════════════════ 277/384 ═══════════════════════════════════════════

/// WHICH CVT INSTRUCTION A NAME IS, WITH ITS ARITY DERIVED FROM THE NAME.
///
/// ⛔⛔ `num_inputs_` AND THE NAME CANNOT DISAGREE HERE. The reference's row carries both as
/// independent fields (`VectorChainHelper.cpp:193-208`) and the scan compares `num_inputs_` with
/// `sources.size()`; the four `sen` sets are already disjoint by arity — binary gcvt is 0/4/24/28
/// against unary 1/2/5/6/8/16/17 — so the variant IS the arity, and the caller's `symbolizeSentientUnary`
/// / `symbolizeSentientBinary` on the returned string is discharged too.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CvtInst {
    /// A `gcvt_imm<n>` a `sentient.vector_unary` takes — one cast source.
    Gcvt1(sen::UnaryGcvt),
    /// An `fcvt_imm<n>` a `sentient.vector_unary` takes — one cast source.
    Fcvt1(sen::UnaryFcvt),
    /// A `gcvt_imm<n>` a `sentient.vector_binary` takes — two cast sources.
    Gcvt2(sen::BinaryGcvt),
    /// An `fcvt_imm<n>` a `sentient.vector_binary` takes — two cast sources.
    Fcvt2(sen::BinaryFcvt),
}

impl CvtInst {
    /// `num_inputs_` — how many `vectorchain.cast` sources feed it.
    #[must_use]
    pub const fn inputs(self) -> usize {
        match self {
            CvtInst::Gcvt1(_) | CvtInst::Fcvt1(_) => 1,
            CvtInst::Gcvt2(_) | CvtInst::Fcvt2(_) => 2,
        }
    }
}

/// THE ONE THING THE SCAN READS OFF A `vectorchain.cast` — its input and result element types.
///
/// ⛔ `llvm::dyn_cast<vectorchain::CastOp>` IS THE CALLER'S, and both call sites do it before
/// building the list (`VectorChainToSentientPESFP.cpp:632`, `:694-697`). `OpBuilder builder(src)`
/// (`VectorChainHelper.cpp:280`) is only `convertTypeToString`'s context and is dropped with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CastSource {
    /// `getElementType(src.getInput().getType())`.
    input: ElemType,
    /// `getElementType(src.getResult().getType())`.
    output: ElemType,
}

impl CastSource {
    /// A cast, or `None` for any other op — the caller's `dyn_cast`.
    #[must_use]
    pub const fn of(op: &vc::Op) -> Option<CastSource> {
        match op {
            vc::Op::Cast { input_ty, ty, .. } => Some(CastSource {
                input: input_ty.elem,
                output: ty.elem,
            }),
            _ => None,
        }
    }
}

/// ONE ROW OF `gcvt_fcvt_insts`.
struct GcvtFcvt {
    /// `name_`, already symbolized — and with it `num_inputs_`.
    name: CvtInst,
    /// `vec_`; `size_` is its length.
    vec: &'static [i32],
    /// `repetition_` — ⭐ every row states it, and `gcvt_imm16`/`17` are the two that are not eight.
    repetition: PackRepetition,
    /// `src_input_type_str_` — the element type EVERY source must convert FROM.
    src_input: TypeSpelling,
    /// `src_output_type_str_` — and TO.
    src_output: TypeSpelling,
}

/// THE EIGHTEEN CVT INSTRUCTIONS, IN THE ORDER THE SCAN WALKS THEM
/// (`VectorChainHelper.cpp:211-267`).
///
/// ⛔⛔ `fcvt_imm4` IS COMMENTED OUT IN THE REFERENCE AND IS THEREFORE ABSENT HERE:
/// *"TODO: Add FCVT mode 4. Need a way to differentiate it from mode 2."* (`:262-264`) — it would
/// have been `{"fcvt_imm4", 2, 8, {0,1,2,3,4,5,6,7}, "fp32", "fp16"}`, row-for-row identical to
/// `fcvt_imm2`, so the scan would never reach it. [`sen::BinaryFcvt::Imm4`] exists and nothing here
/// produces it.
fn gcvt_fcvt_insts() -> [GcvtFcvt; 18] {
    use PackRepetition::{Eight, Four};
    use TypeSpelling::{Bf16, F8E4M3Fn, F8E5M2, Fp16, Fp32, I4, I16};

    const IDENTITY16: &[i32] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    const LOW8: &[i32] = &[0, 1, 2, 3, 4, 5, 6, 7];
    const HIGH8: &[i32] = &[8, 9, 10, 11, 12, 13, 14, 15];
    const INTERLEAVE16: &[i32] = &[0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15];
    const LOW4: &[i32] = &[0, 1, 2, 3];
    const HIGH4: &[i32] = &[4, 5, 6, 7];
    // ⭐ `fcvt_imm3` IS THE ONE PADDED ROW: eight lanes written, eight left alone.
    const LOW8_PADDED: &[i32] = &[0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1];

    let row = |name, repetition, vec, src_input, src_output| GcvtFcvt {
        name,
        vec,
        repetition,
        src_input,
        src_output,
    };
    [
        row(
            CvtInst::Gcvt2(sen::BinaryGcvt::Imm0),
            Eight,
            IDENTITY16,
            Fp16,
            F8E5M2,
        ),
        row(
            CvtInst::Gcvt1(sen::UnaryGcvt::Imm1),
            Eight,
            LOW8,
            F8E5M2,
            Fp16,
        ),
        row(
            CvtInst::Gcvt1(sen::UnaryGcvt::Imm2),
            Eight,
            HIGH8,
            F8E5M2,
            Fp16,
        ),
        row(
            CvtInst::Gcvt2(sen::BinaryGcvt::Imm4),
            Eight,
            IDENTITY16,
            Fp16,
            F8E4M3Fn,
        ),
        row(
            CvtInst::Gcvt1(sen::UnaryGcvt::Imm5),
            Eight,
            LOW8,
            F8E4M3Fn,
            Fp16,
        ),
        row(
            CvtInst::Gcvt1(sen::UnaryGcvt::Imm6),
            Eight,
            HIGH8,
            F8E4M3Fn,
            Fp16,
        ),
        row(CvtInst::Gcvt1(sen::UnaryGcvt::Imm8), Eight, LOW8, I4, I16),
        row(
            CvtInst::Gcvt1(sen::UnaryGcvt::Imm16),
            Four,
            IDENTITY16,
            Fp16,
            Bf16,
        ),
        row(
            CvtInst::Gcvt1(sen::UnaryGcvt::Imm17),
            Four,
            IDENTITY16,
            Bf16,
            Fp16,
        ),
        row(
            CvtInst::Gcvt2(sen::BinaryGcvt::Imm24),
            Eight,
            INTERLEAVE16,
            Fp16,
            F8E5M2,
        ),
        row(
            CvtInst::Gcvt2(sen::BinaryGcvt::Imm28),
            Eight,
            INTERLEAVE16,
            Fp16,
            F8E4M3Fn,
        ),
        row(
            CvtInst::Fcvt1(sen::UnaryFcvt::Imm0),
            Eight,
            LOW4,
            Fp16,
            Fp32,
        ),
        row(
            CvtInst::Fcvt1(sen::UnaryFcvt::Imm1),
            Eight,
            HIGH4,
            Fp16,
            Fp32,
        ),
        row(
            CvtInst::Fcvt2(sen::BinaryFcvt::Imm2),
            Eight,
            LOW8,
            Fp32,
            Fp16,
        ),
        row(
            CvtInst::Fcvt2(sen::BinaryFcvt::Imm3),
            Eight,
            LOW8_PADDED,
            Fp32,
            F8E5M2,
        ),
        row(
            CvtInst::Fcvt1(sen::UnaryFcvt::Imm5),
            Eight,
            LOW4,
            Bf16,
            Fp32,
        ),
        row(
            CvtInst::Fcvt1(sen::UnaryFcvt::Imm6),
            Eight,
            HIGH4,
            Bf16,
            Fp32,
        ),
        row(
            CvtInst::Fcvt2(sen::BinaryFcvt::Imm7),
            Eight,
            LOW8,
            Fp32,
            Bf16,
        ),
    ]
}

/// Replaces: e277_getGCVTorFCVTTypeFromIndicesAndCastInputs
///
/// **277/384** `vectorchain::getGCVTorFCVTTypeFromIndicesAndCastInputs` —
/// `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:188` (108L).
///
/// Which CVT a permutation over cast sources is, or `None` for the caller's *"There is no GCVT
/// instruction corresponding to the following operation"* (`VectorChainToSentientPESFP.cpp:643`).
///
/// ⛔ THE TYPE PAIR IS THE ROW'S, NOT THE SOURCE'S: the loop at `:279-295` compares EVERY source
/// against the SAME `src_input_type_str_`/`src_output_type_str_`, so a two-source pack whose casts
/// convert differently matches nothing. ⛔ AND AN EMPTY `sources` MATCHES NO ROW — arity is 1 or 2.
/// ⭐ `dcc_ext_ctx` is a dropped parameter; the body never reads it.
#[must_use]
pub fn gcvt_or_fcvt_type_from_indices_and_cast_inputs(
    indices: &ValidPackIndices<'_>,
    sources: &[CastSource],
) -> Option<CvtInst> {
    let repetition = indices.repetition();
    let indices = indices.indices();

    for instr in gcvt_fcvt_insts() {
        // `bool found = true;` and the three `continue`s.
        let mut found = true;
        if indices.len() != instr.vec.len() {
            continue;
        }
        if repetition != instr.repetition {
            continue;
        }
        if instr.name.inputs() != sources.len() {
            continue;
        }

        // `for (int i = 0; i < indices.size() && found; i++) if (indices[i] != instr.vec_[i])` —
        // the lengths are equal by the test above, so this is elementwise equality.
        if indices != instr.vec {
            found = false;
        }

        // ⛔ NO `&& found` GUARD ON THIS LOOP in the reference either — it runs over every source
        // and can only clear `found`.
        for src in sources {
            if convert_type_to_string(src.input) != Some(instr.src_input)
                || convert_type_to_string(src.output) != Some(instr.src_output)
            {
                found = false;
            }
        }

        // `if (found) return instr.name_;`
        if found {
            return Some(instr.name);
        }
    }

    // `return "";`
    None
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 340/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e340_analyzeAndFillOperandForwarding
///
/// **340/384** `VectorChainHelper.cpp:535` (25L): every port an operand is ALSO forwarded to, read
/// off the uses of the op that produced it.
///
/// ⛔ THE SPLAT PORT IS PUSHED BEFORE THE WALK and is a use of nothing.
/// ⛔ THE FIVE SKIPPED CLASSES FOLD INTO THE OPERAND ([`is_intermediate`]), so a cast sitting
/// between producer and consumer contributes no port of its own.
pub fn analyze_and_fill_operand_forwarding<A: Arch>(
    comp: ComputeComp,
    from_operand: Option<&VectorOperand>,
    scope: &[DfirOp],
    op_forwarding: &mut Vec<sen::Port>,
) {
    // `if (from_operand.has_value()) {`
    let Some(from) = from_operand else {
        return;
    };

    // `if (is_any_of(from_operand.value().type_, NFWD, ConstantBitstream)) return;`
    if matches!(
        from.kind,
        VectorOperandType::Nfwd | VectorOperandType::ConstantBitstream
    ) {
        return;
    }

    // `if (!from_operand.value().splat_.empty()) op_forwarding.push_back(…splat_…);`
    if let Some(splat) = from.splat {
        op_forwarding.push(splat);
    }

    // `for (auto& use : from_operand.value().op_->getUses()) { Operation* user = use.getOwner();`
    for user in use_positions(&from.op, scope) {
        // `if (!isa<NegOp, CastOp, SelectOp, SIToFPOp, FPToSIOp>(user)) {`
        if is_intermediate(&user, scope) {
            continue;
        }
        // `auto to = VectorOperand::getOperand(dcc_ext_ctx, user, comp, false);`
        if let Some(to) = VectorOperand::operand::<A>(&user, comp, false, scope)
            // `symbolizeSentientComputePort(to.value().getName()).value()` — the empty optional is
            // a value here and an abort there; see [`VectorOperand::name`].
            && let Some(port) = to.name()
        {
            op_forwarding.push(port);
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 341/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHAT ANALYSING ONE NON-COMPUTE OP FOR FUSION ANSWERS — the three by-reference outputs of
/// `analyzeNonComputeOpsForFusion` (`VectorChainHelper.cpp:610-616`), which its callers thread
/// through one op after another.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionAnalysis {
    /// `to_operands` — every destination the fused compute writes.
    ///
    /// ⛔ AN ENTRY MAY BE `None`: the element type is `std::optional<VectorOperand>` and entry 342
    /// pushes an absent one, so the absence is part of the list rather than a gap in it.
    pub to_operands: Vec<Option<VectorOperand>>,
    /// `is_fusion_respected` — cleared by a user that is neither a send nor a store, which is what
    /// stops the caller erasing the from-op (`VectorChainToSentientPESFP.cpp:114`).
    pub is_fusion_respected: bool,
    /// `is_dangling_ops_present_after_fusion` — a store that could not merge, or a user outside the
    /// unit the reuse information was built for.
    pub is_dangling_ops_present_after_fusion: bool,
}

impl Default for FusionAnalysis {
    /// ⛔ `is_fusion_respected` STARTS **TRUE**, which is what both callers write above the call
    /// (`VectorChainToSentientPESFP.cpp:105-107`, `:1219-1221`) — this function only ever clears it,
    /// so a `false` default would refuse every fusion.
    fn default() -> FusionAnalysis {
        FusionAnalysis {
            to_operands: Vec::new(),
            is_fusion_respected: true,
            is_dangling_ops_present_after_fusion: false,
        }
    }
}

/// Replaces: e341_analyzeNonComputeOpsForFusion
///
/// **341/384** `VectorChainHelper.cpp:610` (86L): every send or store the from-op feeds that may
/// fuse into one compute, plus the two flags the caller reads to decide whether the from-op may go.
///
/// ⛔ THE REFERENCE SORTS BY `dominates`, WHICH IS REFLEXIVE and therefore UB on a duplicate. The
/// positions are ordered here instead — lexicographic [`OpId`] order is total and contains
/// dominance, the PROPER form [`OperandReuse::dominates`] says this caller needs.
/// ⛔ A SECOND STORE IS NOT MERGED, IT IS DANGLING; a user that is neither send nor store, and an
/// operand that does not resolve at all, both clear `is_fusion_respected`.
pub fn analyze_non_compute_ops_for_fusion<A: Arch>(
    unit: &OpId,
    op: &OpId,
    from: &VectorOperand,
    comp: ComputeComp,
    is_visited: &mut BTreeMap<OpId, bool>,
    scope: &[DfirOp],
    into: &mut FusionAnalysis,
) {
    // `auto to_operand = VectorOperand::getOperand(dcc_ext_ctx, op, comp, false);`
    let to_operand = VectorOperand::operand::<A>(op, comp, false, scope);

    // `if (to_operand.has_value() && isa<SendOp, vector::StoreOp, agen::VectorStoreOp>(…op_))`
    if let Some(to) = &to_operand
        && is_send_or_store(&to.op, scope)
    {
        into.to_operands.push(to_operand.clone());
    }

    // `bool is_store_op_encountered = isa<agen::VectorStoreOp, vector::StoreOp>(to_operand.value().op_);`
    // ⛔ THAT `.value()` IS UNGUARDED one line under a `has_value()` test, so an unresolved operand
    // aborts the reference here. `false` is the total answer, and it is the one the guarded line
    // above already took.
    let mut is_store_op_encountered = to_operand
        .as_ref()
        .is_some_and(|to| is_store(&to.op, scope));

    // The two `getUsers()` arms. ⭐ THE CONSTANT ONE CONFINES THEM TO `unit` because the reuse
    // information is per unit and a hoisted constant is read from outside it; the other takes every
    // user. `if (user)` is a null test this island has no case for.
    let from_is_constant = is_arith_constant(&from.op, scope);
    let mut users: Vec<OpId> = Vec::new();
    for user in use_positions(&from.op, scope) {
        if !from_is_constant || user.path().starts_with(unit.path()) {
            users.push(user);
        } else {
            into.is_dangling_ops_present_after_fusion = true;
        }
    }

    // `llvm::sort(users, [&](l, r) { return reuse_info.dominates(l, r); });` — see the note above on
    // why the comparator is the positions' own order and why `reuse_info` carries nothing here.
    users.sort();

    // `// TODO: what happens when you have two sends from a from-op.`
    for user in users {
        // `auto to = VectorOperand::getOperand(dcc_ext_ctx, user, comp, false);`
        let to = VectorOperand::operand::<A>(&user, comp, false, scope);
        match &to {
            // `if (to.has_value() && to.value().op_ != op && !is_visited[to.value().op_]) {`
            Some(found)
                if &found.op != op && !is_visited.get(&found.op).copied().unwrap_or(false) =>
            {
                // `bool allow_merging = isa<SendOp, vector::StoreOp, agen::VectorStoreOp>(…);`
                if is_send_or_store(&found.op, scope) {
                    let store = is_store(&found.op, scope);
                    // `allow_merging = !(isa<vector::StoreOp, agen::VectorStoreOp>(…) &&
                    //  is_store_op_encountered);`
                    let allow_merging = !(store && is_store_op_encountered);
                    // ⛔ SET AFTER THE TEST AND WHETHER OR NOT IT PASSED, so the first store merges
                    // and every later one dangles.
                    if store {
                        is_store_op_encountered = true;
                    }
                    if allow_merging {
                        into.to_operands.push(to.clone());
                        is_visited.insert(found.op.clone(), true);
                    } else {
                        into.is_dangling_ops_present_after_fusion = true;
                    }
                } else {
                    into.is_fusion_respected = false;
                }
            }
            // `else if (!to.has_value()) is_fusion_respected = false;`
            None => into.is_fusion_respected = false,
            // The current op and an already-visited destination fall through setting nothing.
            Some(_) => {}
        }
    }
}

/// WHETHER THE OP AT A POSITION IS A STORE — `isa<agen::VectorStoreOp, vector::StoreOp>`
/// (`VectorChainHelper.cpp:632-633`).
fn is_store(id: &OpId, scope: &[DfirOp]) -> bool {
    matches!(
        op_at(id, scope),
        Some(
            DfirOp::Agen(dfir_op::agen::Op::VectorStore { .. })
                | DfirOp::Vector(dfir_op::vector::Op::Store { .. })
        )
    )
}

/// WHETHER IT IS A SEND OR A STORE — `isa<dataflow::SendOp, vector::StoreOp, agen::VectorStoreOp>`
/// (`VectorChainHelper.cpp:618-619`, `:672-674`), the class that may fuse into a compute.
fn is_send_or_store(id: &OpId, scope: &[DfirOp]) -> bool {
    matches!(
        op_at(id, scope),
        Some(DfirOp::Dataflow(dfir_op::dataflow::Op::Send { .. }))
    ) || is_store(id, scope)
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 342/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e342_analyzeAndFillResultForwarding
///
/// **342/384** `VectorChainHelper.hpp:162` (27L): where a compute's result goes — one port per
/// user, with an `istate` destination split off as the LOGICAL result.
///
/// ⛔ AN `sfpring` DESTINATION ALSO EMITS `sentient.set_send_dst` NAMING THE SEND'S OWN UNIT,
/// stepping over an intervening cast to find that send. Positioning it after the send is the
/// caller's job here.
/// ⛔ `to_operands` KEEPS AN ABSENT OPERAND, which the reference then dereferences.
pub fn analyze_and_fill_result_forwarding<A: Arch>(
    op: &OpId,
    comp: ComputeComp,
    scope: &[DfirOp],
    to_operands: &mut Vec<Option<VectorOperand>>,
    result_forwarding: &mut Vec<sen::Port>,
    logical_result_forwarding: &mut Option<sen::Port>,
    set_send_dst: &mut Vec<SenOp>,
) {
    // `for (auto user : op->getUsers()) {`
    for user in use_positions(op, scope) {
        // `auto to = …getOperand(dcc_ext_ctx, user, comp, false); to_operands.push_back(to);`
        let to = VectorOperand::operand::<A>(&user, comp, false, scope);
        to_operands.push(to.clone());

        // `std::string dest = to.value().getName();` — unguarded there; see [`VectorOperand::name`].
        let Some(dest) = to.and_then(|to| to.name()) else {
            continue;
        };

        // `if (dest == "sfpring") {`
        if dest == sen::Port::SfpRing {
            // `if (isa<FPToSIOp, SIToFPOp, vectorchain::CastOp>(user)) {
            //    DT_CHECK_MSG(!user->getUsers().empty(), …); user = *user->getUsers().begin(); }`
            let sending = if is_cast(&user, scope) {
                use_positions(&user, scope).into_iter().next()
            } else {
                Some(user.clone())
            };
            // `auto send_op = dyn_cast<dataflow::SendOp>(user); builder.setInsertionPointAfter(…);`
            // `sentient::SetSendDestinationOp::create(builder, op->getLoc(), send_op.getToUnit());`
            if let Some(sending) = sending
                && let Some(DfirOp::Dataflow(dfir_op::dataflow::Op::Send { to, .. })) =
                    op_at(&sending, scope)
            {
                set_send_dst.push(SenOp::Sentient(sen::Op::SetSendDst { units: *to }));
            }
        }

        // `if (dest.find("istate") != dest.npos) logical_result_forwarding = …; else …push_back(…);`
        if matches!(dest, sen::Port::IState(_)) {
            *logical_result_forwarding = Some(dest);
        } else {
            result_forwarding.push(dest);
        }
    }
}

/// WHETHER THE OP AT A POSITION IS ONE OF THE THREE CONVERSIONS THAT SIT BETWEEN A COMPUTE AND ITS
/// SEND — `isa<arith::FPToSIOp, arith::SIToFPOp, vectorchain::CastOp>` (`VectorChainHelper.hpp:175`).
///
/// ⛔ THREE CLASSES AND NOT [`is_intermediate`]'S FIVE: a `neg` or a `select` here is NOT stepped
/// over, so its own port is the destination.
fn is_cast(id: &OpId, scope: &[DfirOp]) -> bool {
    matches!(
        op_at(id, scope),
        Some(
            DfirOp::Arith(dfir_op::arith::Op::FpToSi(_) | dfir_op::arith::Op::SiToFp(_))
                | DfirOp::VectorChain(vc::Op::Cast { .. })
        )
    )
}

#[cfg(test)]
mod unit_tests {
    use super::{CastSource, CvtInst, gcvt_or_fcvt_type_from_indices_and_cast_inputs};
    use super::{DfirProgram, SenOp, SentientProgramUnit, Values};
    use super::{
        MergeAndPack, MergeOrPack, MinOrMaxFusion, NonPtMask, OperandReuse, PackOrShuffle,
        PackRepetition, SliceMask, TypeSpelling, check_validity_of_pack_and_shuffle_lowering,
        compute_precision_of_op, convert_string_to_type, convert_type_to_string,
        fuse_compare_and_select_into_min_or_max, get_mask_value_constant_for_non_pt,
        get_mask_value_for_non_pt, has_constant_bounds, input_precision_from_operand,
        is_sentient_binary_logical_op, merge_and_pack_insts, merge_type_from_indices,
        precision_in_string, redefine_constant_vectors, reset_sentient_fmas_if_exists,
        result_precision_from_operands, validate_lowering_and_set_missing_parameters,
        vector_binary_to_sentient_binary,
        vector_element_wise_compare_operator_to_sentient_binary_operator,
        vector_ternary_to_sentient_ternary, vector_type_of,
    };
    use crate::arch::Dd2;
    use crate::formats::Bits;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::link::{Link, Lxlu, Lxsu, Sfp};
    use crate::islands::dataflow_ir::ty::MemRef;
    use crate::islands::dataflow_ir::{
        Grid, GroupId, OpIndex, ProgramName, ProgramUnit as DfirProgramUnit, ProgramUnits,
    };

    /// The arch every fixture below is built for; nothing here is arch-dependent.
    type Target = Dd2;

    use super::{
        FusionAnalysis, analyze_and_fill_operand_forwarding, analyze_and_fill_result_forwarding,
        analyze_non_compute_ops_for_fusion,
    };
    use crate::bridges::dataflow_ir_to_sentient::vc_vector_operands::{
        ComputeComp, OpId, VectorOperand, VectorOperandType,
    };
    use crate::islands::dataflow_ir::dialects::Val;
    use crate::islands::dataflow_ir::dialects::vectorchain as vc;
    use crate::islands::dataflow_ir::dialects::{self as dfir_op, Op as DfirOp};
    use crate::islands::dataflow_ir::ty::{
        AffineExpr, AffineMap, Constraint, ElemType, IntegerSet, ScalarTy, Vector,
    };
    use crate::islands::sentient::dialects::sentient as sen;
    use crate::units::{DfirUnit, Residency};
    use std::collections::BTreeMap;

    /// An operand off the link with the two precisions under test.
    fn operand(orig: Option<sen::Precision>, on_the_fly: Option<sen::Precision>) -> VectorOperand {
        VectorOperand {
            kind: VectorOperandType::Link,
            op: OpId::at(&[0]),
            values: Vec::new(),
            orig_precision: orig,
            on_the_fly_conv_precision: on_the_fly,
            splat: None,
        }
    }

    /// ⭐ NO VENDOR CASE TO PORT: `dcc/test` reaches this only through whole-program
    /// `CHECK-SENT-IR`, where its answer shows up as which `result_precision` a chain of `and0`s
    /// ends up carrying — not as a value of its own.
    #[test]
    fn the_four_bitwise_operators_are_logical() {
        for sb in [
            sen::BinaryOp::And,
            sen::BinaryOp::Or,
            sen::BinaryOp::Xnor,
            sen::BinaryOp::AndNot,
        ] {
            assert!(is_sentient_binary_logical_op(sb), "{sb:?}");
        }
    }

    #[test]
    fn the_arithmetic_and_convert_operators_are_not() {
        for sb in [
            sen::BinaryOp::Min,
            sen::BinaryOp::Max,
            sen::BinaryOp::AbsMin,
            sen::BinaryOp::AbsMax,
            sen::BinaryOp::Add,
            sen::BinaryOp::Mul,
            sen::BinaryOp::Sub,
            sen::BinaryOp::MulDiv2,
            sen::BinaryOp::GcvtImm(sen::BinaryGcvt::Imm0),
            sen::BinaryOp::FcvtImm(sen::BinaryFcvt::Imm2),
        ] {
            assert!(!is_sentient_binary_logical_op(sb), "{sb:?}");
        }
    }

    /// ⛔ A `merge`, A `pack` AND A COMPARE ARE NOT LOGICAL EITHER, however bit-twiddling they look.
    /// `merge8l` shuffles bytes and `fcmp_eq` yields a mask, and neither is in the reference's
    /// four-element list.
    #[test]
    fn the_merges_packs_and_compares_are_not() {
        for sb in [
            sen::BinaryOp::Merge {
                width: sen::MergeWidth::W8,
                high: false,
            },
            sen::BinaryOp::Pack(sen::PackIndex::P0),
            sen::BinaryOp::CompareNeq,
            sen::BinaryOp::CompareEq,
            sen::BinaryOp::CompareLt,
            sen::BinaryOp::CompareLe,
        ] {
            assert!(!is_sentient_binary_logical_op(sb), "{sb:?}");
        }
    }

    /// ⭐ 060/384 — IT READS `orig_precision_`, and it does not read the other one.
    #[test]
    fn the_input_precision_is_the_operands_original_one() {
        let opr = operand(Some(sen::Precision::Int8), Some(sen::Precision::Fp16));
        assert_eq!(
            input_precision_from_operand(&opr),
            Some(sen::Precision::Int8),
            "`on_the_fly_conv_precision_` is the result precision's field, not this one"
        );
    }

    /// ⭐ 060/384 — AND AN OPERAND NOTHING SET A PRECISION ON ANSWERS ABSENT, which is the C++'s
    /// `""` and NOT `Precision::None`.
    #[test]
    fn an_operand_with_no_precision_answers_absent() {
        assert_eq!(input_precision_from_operand(&operand(None, None)), None);
    }

    /// ⭐ 060/384 — THE `std::optional` OVERLOAD (`:44-50`) IS `and_then` OVER THIS ONE.
    #[test]
    fn the_optional_overload_is_and_then_over_the_reference_one() {
        let present = Some(operand(Some(sen::Precision::Fp8), None));
        let absent: Option<VectorOperand> = None;
        assert_eq!(
            present.as_ref().and_then(input_precision_from_operand),
            Some(sen::Precision::Fp8)
        );
        assert_eq!(absent.as_ref().and_then(input_precision_from_operand), None);
    }

    /// ⭐ 061/384 — THE FIRST PRESENT OPERAND'S ON-THE-FLY PRECISION, past any `nullopt` before it.
    #[test]
    fn the_result_precision_skips_absent_operands() {
        let operands = [
            None,
            None,
            Some(operand(
                Some(sen::Precision::Int8),
                Some(sen::Precision::Fp16),
            )),
            Some(operand(None, Some(sen::Precision::Fp32))),
        ];
        assert_eq!(
            result_precision_from_operands(&operands),
            Some(sen::Precision::Fp16)
        );
    }

    /// ⛔ 061/384 — A FIRST OPERAND THAT EXISTS WITH NO PRECISION ENDS THE SEARCH.
    ///
    /// This is the case a `filter_map(..).next()` gets wrong: the `return` sits inside
    /// `if (opr_.has_value())` and fires whatever the string turned out to be
    /// (`VectorChainHelper.cpp:52-62`), so the `Fp32` behind it is never reached and
    /// `validateLoweringAndSetMissingParameters` gets the empty answer it substitutes the compute
    /// precision for.
    #[test]
    fn a_present_operand_with_no_precision_ends_the_search() {
        let operands = [
            Some(operand(Some(sen::Precision::Int8), None)),
            Some(operand(None, Some(sen::Precision::Fp32))),
        ];
        assert_eq!(
            result_precision_from_operands(&operands),
            None,
            "the reference returns on the first PRESENT operand, not on the first precision"
        );
    }

    /// ⭐ 061/384 — AND AN ALL-ABSENT LIST ANSWERS ABSENT, which is the `return "";` the extract
    /// truncated away.
    #[test]
    fn a_list_with_no_operands_at_all_answers_absent() {
        assert_eq!(result_precision_from_operands(&[None, None]), None);
        assert_eq!(result_precision_from_operands(&[]), None);
    }

    /// A vector of `len` elements of `elem`.
    fn vec(len: u64, elem: ElemType) -> Vector {
        Vector { len, elem }
    }

    /// `vectorchain.shuffle` producing `ty` — the op two of IBM's three cases below reach
    /// [`compute_precision_of_op`] through.
    fn shuffle(ty: Vector) -> DfirOp {
        DfirOp::VectorChain(vc::Op::Shuffle {
            result: Val(44),
            input: Val(43),
            variable: Vec::new(),
            pad: Vec::new(),
            mask: None,
            indices: (0..16).collect(),
            repetition: 4,
            input_ty: ty,
            ty,
        })
    }

    /// ⭐⭐ 062/384 — IBM'S OWN ANSWER ON A PACK: `fp16` WHERE THE ELEMENT TYPE SAYS `fp8`.
    ///
    /// `dcc/test/Conversion/VectorChainToSentientPESFP/gcvt.mlir:77` packs two
    /// `vector<128xf8E4M3FN>` operands, and the compute it lowers to carries
    /// `ComputePrecision = #sentient<precision fp16>` beside `ResultPrecision = fp8` in the same
    /// file's `CHECK-SENT-IR`. So this is not a shortcut past the element type — it is a DIFFERENT
    /// answer from the one the element type gives, and the only test that can tell them apart is one
    /// where they disagree.
    #[test]
    fn a_pack_computes_in_fp16_whatever_its_elements_are() {
        let fp8x128 = vec(128, ElemType::F8E4M3Fn);
        let pack = DfirOp::VectorChain(vc::Op::Pack {
            result: Val(16),
            op1: Val(14),
            op2: Val(15),
            mask: None,
            indices: (0..16).collect(),
            repetition: 8,
            sign_extend: false,
            operand_ty: fp8x128,
            ty: fp8x128,
        });
        assert_eq!(compute_precision_of_op(&pack), sen::Precision::Fp16);
        assert_eq!(
            precision_in_string(ElemType::F8E4M3Fn),
            sen::Precision::Fp8,
            "and the element type really does say fp8, so the two answers differ"
        );
    }

    /// ⭐⭐ 062/384 — AND ON A MERGE, WITHOUT ASKING FOR AN ELEMENT TYPE AT ALL.
    ///
    /// ⛔ THE ORDER IS WHAT IS UNDER TEST. `MergeOp` is in neither `getVectorType` nor
    /// `getCustomVectorType`, so [`vector_type_of`] answers `None` for it and an element-type query
    /// would abort — which is exactly what the reference's `DT_CHECK_MSG` does. The short-circuit
    /// running first is the difference between a compiled program and a crash.
    #[test]
    fn a_merge_computes_in_fp16_and_has_no_element_type_to_ask_for() {
        let bf16x64 = vec(64, ElemType::Bf16);
        let merge = DfirOp::VectorChain(vc::Op::Merge {
            result: Val(3),
            op1: Val(1),
            op2: Val(2),
            iteration_space_ca: IntegerSet::from_sizes(&[64]),
            access_function_ca: AffineMap::identity(1),
            access_function_a: AffineMap::identity(1),
            iteration_space_cb: IntegerSet::from_sizes(&[64]),
            access_function_cb: AffineMap::identity(1),
            access_function_b: AffineMap::identity(1),
            operand_ty: bf16x64,
            ty: bf16x64,
        });
        assert_eq!(compute_precision_of_op(&merge), sen::Precision::Fp16);
        assert_eq!(
            vector_type_of(&merge),
            None,
            "so reaching getElementType for a merge is the abort the short-circuit avoids"
        );
    }

    /// ⭐⭐ 062/384 — IBM'S OWN ANSWER ON A `bf16` OP: `fp16`, THE REMAP.
    ///
    /// `dcc/test/Conversion/VectorChainToSentientPESFP/gcvt_sen1p5.mlir:139` shuffles
    /// `vector<64xbf16>` into `vector<64xbf16>`, and the `sentient.vector_unary` it lowers to carries
    /// `ComputePrecision = #sentient<precision fp16>, … ResultPrecision = #sentient<precision bf16>`
    /// (that file's `CHECK-SENT-IR`, line 51). The RESULT keeps `bf16` and the COMPUTE does not —
    /// which is the whole content of `is_any_of(precision, "bf16", "dlfp16")`.
    #[test]
    fn a_bf16_op_computes_in_fp16_while_its_result_stays_bf16() {
        assert_eq!(
            compute_precision_of_op(&shuffle(vec(64, ElemType::Bf16))),
            sen::Precision::Fp16
        );
        assert_eq!(
            precision_in_string(ElemType::Bf16),
            sen::Precision::Bf16,
            "getPrecisionInString itself does NOT remap — the caller does"
        );
    }

    /// ⭐⭐ 062/384 — AND AN `i16` OP COMPUTES IN `int16`, unremapped.
    ///
    /// `gcvt_sen1p5.mlir:133` shuffles `vector<256xi16>` into `vector<64xi16>`, and its
    /// `sentient.vector_unary` carries `ComputePrecision = #sentient<precision int16>` (line 49) —
    /// the case that proves the element-type route runs at all, rather than everything arriving at
    /// `fp16` by some other means.
    #[test]
    fn an_i16_op_computes_in_int16() {
        assert_eq!(
            compute_precision_of_op(&shuffle(vec(64, ElemType::Int(16)))),
            sen::Precision::Int16
        );
    }

    /// ⛔ 062/384 — `f16` IS `fp16`, NOT `ieee_fp16`.
    ///
    /// `getPrecisionInString`'s float branch is `"fp" + bitwidth`, so `ieee_fp16` — a real
    /// [`sen::Precision`] — is unreachable from an element type. IBM's `gcvt_sen1p5.mlir:115`
    /// shuffles `vector<128xf16>` and the compute reads `fp16`.
    #[test]
    fn an_f16_op_computes_in_fp16_and_never_in_ieee_fp16() {
        assert_eq!(
            compute_precision_of_op(&shuffle(vec(64, ElemType::F16))),
            sen::Precision::Fp16
        );
        assert_ne!(precision_in_string(ElemType::F16), sen::Precision::IeeeFp16);
    }

    /// ⛔ 062/384 — THE FIVE VECTOR-RESULTED OPS THAT ARE STILL OUTSIDE THE ELEMENT-TYPE DOMAIN.
    ///
    /// A cast, a select, a constant bitstream, a lane mask and a merge all bind a vector, and the
    /// reference lists none of them in `getVectorType` or `getCustomVectorType` — so
    /// `getElementType` aborts on each. Answering from "it obviously has a result type" would hide
    /// the abort that `PackOpLowering`'s cast check is written around.
    #[test]
    fn a_cast_and_a_select_are_outside_the_element_type_domain() {
        let f16x64 = vec(64, ElemType::F16);
        let cast = DfirOp::VectorChain(vc::Op::Cast {
            result: Val(43),
            input: Val(42),
            input_ty: f16x64,
            ty: vec(64, ElemType::Bf16),
        });
        let select = DfirOp::VectorChain(vc::Op::Select {
            result: Val(51),
            input: Val(50),
            selection_map: AffineMap::identity(1),
            input_ty: f16x64,
            ty: f16x64,
        });
        assert_eq!(vector_type_of(&cast), None);
        assert_eq!(vector_type_of(&select), None);
    }

    /// ⭐ 062/384 — AND THE OPS THAT ARE IN THE DOMAIN ANSWER WITH THE TYPE THEY CARRY.
    ///
    /// A send, a receive and an `agen` load are the three non-`vectorchain` members of the reference's
    /// lists that this island can spell; without them the element-type route would be
    /// `vectorchain`-only and every compute precision would come from a compute rather than from the
    /// data that reached it.
    #[test]
    fn a_send_and_an_agen_load_carry_their_vector_type() {
        let i8x128 = vec(128, ElemType::Int(8));
        assert_eq!(
            vector_type_of(&shuffle(i8x128)),
            Some(i8x128),
            "a shuffle is in the list"
        );
        let program = DfirOp::Arith(dfir_op::arith::Op::ConstantInt {
            result: Val(0),
            value: dfir_op::arith::IntConst::Bool(true),
        });
        assert_eq!(
            vector_type_of(&program),
            None,
            "and neither reference list mentions an arith op"
        );
    }

    /// ⭐⭐ 063/384 — EVERY ONE-DIMENSIONAL MASK SET IN THE AUTHORITY TREE PASSES THIS GATE.
    ///
    /// The five distinct shapes its `.mlir` tests write, counted across `dcc/test`: lanes `0..=63`
    /// (22 files), `0..=127` (12), `0..=31` (10), the single lane `0..=0` (12), and the CROSSING
    /// `64..=63` (13). All five must answer `true` — the last one is the all-lanes-off mask, and a
    /// `Lb <= Ub` test here would reject it.
    #[test]
    fn every_one_dimensional_mask_set_ibm_writes_has_constant_bounds() {
        for (from, to) in [(0, 63), (0, 127), (0, 31), (0, 0), (64, 63)] {
            let mask = IntegerSet {
                dims: 1,
                symbols: 0,
                constraints: vec![
                    Constraint {
                        expr: AffineExpr::dim(0).plus(AffineExpr::Const(-from)),
                        is_equality: false,
                    },
                    Constraint {
                        expr: AffineExpr::dim(0).times(-1).plus(AffineExpr::Const(to)),
                        is_equality: false,
                    },
                ],
            };
            assert!(has_constant_bounds(&mask), "d0 in {from}..={to}");
        }
    }

    /// ⛔ 063/384 — AND A SET THAT BOUNDS ONE SIDE, OR NEITHER, DOES NOT.
    ///
    /// This is the whole point of the gate: `getMaskValueConstantForNonPT` divides by the bounds it
    /// reads, so a missing one has to stop the lowering rather than default to a lane index.
    #[test]
    fn a_mask_set_missing_either_side_has_no_constant_bounds() {
        let lower_only = IntegerSet {
            dims: 1,
            symbols: 0,
            constraints: vec![Constraint {
                expr: AffineExpr::dim(0),
                is_equality: false,
            }],
        };
        let upper_only = IntegerSet {
            dims: 1,
            symbols: 0,
            constraints: vec![Constraint {
                expr: AffineExpr::dim(0).times(-1).plus(AffineExpr::Const(63)),
                is_equality: false,
            }],
        };
        let unconstrained = IntegerSet {
            dims: 1,
            symbols: 0,
            constraints: vec![],
        };
        assert!(!has_constant_bounds(&lower_only));
        assert!(!has_constant_bounds(&upper_only));
        assert!(!has_constant_bounds(&unconstrained));
    }

    /// ⭐ 063/384 — A PINNED LANE HAS BOUNDS, from its equality alone.
    ///
    /// `affine_set<(d0) : (d0 == 0)>` is `#set1` of a dozen of IBM's files and what
    /// [`IntegerSet::from_sizes`] writes for one lane; the equality answers both sides, so no
    /// inequality is needed.
    #[test]
    fn a_single_pinned_lane_has_constant_bounds() {
        assert!(has_constant_bounds(&IntegerSet::from_sizes(&[1])));
    }

    /// ⭐⭐ 064/384 — `pack24`, IBM'S LONGEST ROW: SIXTY-FOUR INDICES SUMMING TO 912.
    ///
    /// `VectorChainHelper.cpp:328-332` writes sixteen real lanes `0, 8, … 120` followed by
    /// FORTY-EIGHT `-1`s. The lanes alone sum to 960; the row's `sum` is 912, because the `-1`s are
    /// part of it. `getMergeTypeFromIndices` compares that number against the sum of the pack's own
    /// indices (`:400`), so dropping the `-1`s would make the filter accept index lists 48 apart from
    /// this one.
    #[test]
    fn ibms_pack24_row_is_sixty_four_indices_summing_to_912() {
        let mut vec: Vec<i32> = (0..16).map(|lane| lane * 8).collect();
        vec.extend(std::iter::repeat_n(-1, 48));
        let row = MergeAndPack::new(MergeOrPack::Pack(sen::PackIndex::P24), Bits(2), vec, false);
        assert_eq!(row.size(), 64);
        assert_eq!(row.sum(), 912);
        assert_eq!(
            row.vec().iter().filter(|lane| **lane >= 0).sum::<i32>(),
            960,
            "and the sixteen real lanes on their own sum to 960 — the 48 misses are the difference"
        );
    }

    /// ⭐⭐ 064/384 — `pack12` AND `pack14` DIFFER IN NOTHING BUT `sign_extend`.
    ///
    /// `VectorChainHelper.cpp:346` and `:350-353` carry the same eight lanes at the same
    /// `element_bit_width`, so their `size` and `sum` are equal and the matching loop reaches the same
    /// per-lane verdict for both; the LAST check, `inst.sign_extend == sign_extend` (`:409`), is the
    /// only thing that tells them apart. A row that dropped the flag or defaulted it would make one
    /// of the two unreachable and silently emit the other.
    #[test]
    fn pack12_and_pack14_are_separated_only_by_sign_extension() {
        let lanes: Vec<i32> = (0..8).flat_map(|lane| [lane, -1]).collect();
        let pack12 = MergeAndPack::new(
            MergeOrPack::Pack(sen::PackIndex::P12),
            Bits(8),
            lanes.clone(),
            false,
        );
        let pack14 =
            MergeAndPack::new(MergeOrPack::Pack(sen::PackIndex::P14), Bits(8), lanes, true);
        assert_eq!(pack12.size(), pack14.size());
        assert_eq!(pack12.sum(), pack14.sum());
        assert_eq!(pack12.vec(), pack14.vec());
        assert_eq!(pack12.size(), 16);
        assert_eq!(pack12.sum(), 20, "0..=7 is 28, less eight misses");
        assert!(!pack12.sign_extend());
        assert!(pack14.sign_extend());
    }

    /// ⭐ 064/384 — A MERGE ROW, AND ITS NAME IS THE SENTIENT OPERATOR RATHER THAN A STRING.
    ///
    /// `{"merge16h", 16, {1, 9, 3, 11, 5, 13, 7, 15}}` (`:369`) is eight indices summing to 64, and
    /// the spelling `getMergeTypeFromIndices` would return for it is
    /// [`sen::BinaryOp::spelling`]'s — so there is one place that knows how a mnemonic is written.
    #[test]
    fn ibms_merge16h_row_carries_the_operator_not_its_spelling() {
        let row = MergeAndPack::new(
            MergeOrPack::Merge {
                width: sen::MergeWidth::W16,
                high: true,
            },
            Bits(16),
            vec![1, 9, 3, 11, 5, 13, 7, 15],
            false,
        );
        assert_eq!(row.size(), 8);
        assert_eq!(row.sum(), 64);
        assert_eq!(row.element_bit_width(), Bits(16));
        assert_eq!(row.name().as_binary().spelling(), "merge16h");
    }

    /// ⛔ 064/384 — AND `repetition` IS EIGHT WITHOUT ANYONE PASSING IT.
    ///
    /// `int repetition = 8;` is a member default the constructor never assigns and no row of the
    /// table overrides (`:309`), yet `getMergeTypeFromIndices` compares it for equality against the
    /// pack op's own `repetition` (`:399`) — so a zero here would reject every candidate row and
    /// `getMergeTypeFromIndices` would answer "no such instruction" for every legal pack.
    #[test]
    fn every_row_repeats_eight_times() {
        let row = MergeAndPack::new(
            MergeOrPack::Pack(sen::PackIndex::P25),
            Bits(8),
            (0..16).map(|lane| lane * 2).collect(),
            false,
        );
        assert_eq!(row.repetition(), 8);
        assert_eq!(row.size(), 16);
        assert_eq!(row.sum(), 240, "0, 2, .. 30");
    }

    // ═══════════════════ e065-e070 ═══════════════════

    /// The `vector<64xf16>` every case below computes on.
    fn f16x64() -> Vector {
        vec(64, ElemType::F16)
    }

    /// A mask or a compare result — `vector<64xi1>`.
    fn mask_ty() -> Vector {
        vec(64, ElemType::Int(1))
    }

    /// A predicate over `val`, which is how a compare's result reaches a selection's `cond`.
    fn pred(val: u32) -> vc::Predicate {
        vc::LaneMask::prefix_of(64, mask_ty()).binds(Val(val))
    }

    /// `dataflow.receive … : vector<64xf16>` binding `result` — `fmax.mlir:112`.
    fn receive(result: u32) -> DfirOp {
        DfirOp::Dataflow(dfir_op::dataflow::Op::Receive {
            result: Val(result),
            from: Link::<Lxlu, Sfp>::between(Val(90), Val(91)).ends().1,
            ty: f16x64(),
        })
    }

    /// `agen.vector_load %view[0] … : vector<64xf16>` binding `result` — `fmax.mlir:116`.
    fn vector_load(result: u32) -> DfirOp {
        DfirOp::Agen(dfir_op::agen::Op::VectorLoad {
            dbg_name: None,
            result: Val(result),
            view: Val(92),
            indices: std::vec![dfir_op::Index::Const(0)],
            view_ty: MemRef {
                shape: std::vec![64, 4, 1],
                elem: ElemType::F16,
            },
            ty: f16x64(),
            multicast_info: None,
        })
    }

    /// `arith.constant dense<0.000000e+00> : vector<64xf16>` — `relu.mlir:50`.
    fn dense_zero(result: u32) -> DfirOp {
        DfirOp::Arith(dfir_op::arith::Op::DenseConstant {
            result: Val(result),
            splat: 0,
            ty: f16x64(),
        })
    }

    /// `vectorchain.element_wise_compare %op1, %op2[%mask] {compare_op}`.
    fn compare(result: u32, op1: u32, op2: u32, compare_op: vc::CompareOp) -> DfirOp {
        DfirOp::VectorChain(vc::Op::ElementWiseCompare {
            result: Val(result),
            op1: Val(op1),
            op2: Val(op2),
            mask: Some(pred(56)),
            compare_op,
            operand_ty: f16x64(),
            ty: mask_ty(),
        })
    }

    /// 🎯 `fmax.mlir` — A RECEIVE AND A LOAD, COMPARED `gt` AND SELECTED IN THAT ORDER, IS ONE `max`.
    ///
    /// The vendor input is `element_wise_compare %53, %55 {compare_gt}` followed by
    /// `element_wise_selection %57 ? %53 : %55` (`fmax.mlir:119-120`), and the expectation is a single
    /// `sentient.vector_binary … binaryOp = #sentient<binary_operator max>` (`fmax.mlir:42`).
    #[test]
    fn a_receive_and_a_load_compared_greater_than_fuse_to_max() {
        let scope = std::vec![
            receive(53),
            vector_load(55),
            compare(57, 53, 55, vc::CompareOp::Gt),
        ];

        assert_eq!(
            fuse_compare_and_select_into_min_or_max(pred(57), Val(53), Val(55), &scope),
            MinOrMaxFusion::ToMax
        );
    }

    /// 🎯 THE ARMS ARE MATCHED BY STRUCTURE, SO TWO SEPARATE `dense<0.0>` CONSTANTS STILL FUSE.
    ///
    /// ⛔⛔ THIS IS THE CASE THE REFERENCE'S COMMENT EXISTS FOR — *"since some times constant operands
    /// are duplicated, and direct match may result in spurious mismatches"*. `relu.mlir:50-51` defines
    /// `%cst` and `%cst_1` as two distinct `arith.constant dense<0.000000e+00> : vector<64xf16>` ops;
    /// the compare reads the first and the selection the second (`:73-74`), and `relu.mlir:35,37` still
    /// shows the `max` with `opB = #sentient<compute_port zero>`. A `lhs == op1 && rhs == op2` test on
    /// SSA identity answers *false* here and emits two instructions instead of one.
    #[test]
    fn two_separate_but_identical_zero_constants_still_fuse() {
        let scope = std::vec![
            receive(16),
            dense_zero(100),
            dense_zero(101),
            compare(18, 16, 100, vc::CompareOp::Gt),
        ];

        assert_eq!(
            fuse_compare_and_select_into_min_or_max(pred(18), Val(16), Val(101), &scope),
            MinOrMaxFusion::ToMax,
            "%cst_1 is a different value from %cst but the same computation"
        );
    }

    /// 🎯 THE FOUR ORDERING COMPARISONS AGAINST THE TWO ARM ORDERS — the reference's own table
    /// (`VectorChainHelper.cpp:444-459`).
    ///
    /// ⭐ SWAPPING THE SELECTION'S ARMS INVERTS THE ANSWER, which is the whole reason `same` and
    /// `opposite` are tracked separately: `a > b ? a : b` is `max`, and `a > b ? b : a` is `min`.
    ///
    /// ⚠️ NO VENDOR CASE PRODUCES A `min` — only three `.mlir` files in the authority tree contain an
    /// `element_wise_selection` at all and all three fuse to `max` or not at all. The three rows below
    /// that no vendor test pins come from the reference's branch table, and are marked as such.
    #[test]
    fn the_arm_order_decides_between_min_and_max() {
        let table = std::vec![
            (vc::CompareOp::Gt, true, MinOrMaxFusion::ToMax),
            (vc::CompareOp::Ge, true, MinOrMaxFusion::ToMax),
            (vc::CompareOp::Gt, false, MinOrMaxFusion::ToMin),
            (vc::CompareOp::Ge, false, MinOrMaxFusion::ToMin),
            (vc::CompareOp::Lt, true, MinOrMaxFusion::ToMin),
            (vc::CompareOp::Le, true, MinOrMaxFusion::ToMin),
            (vc::CompareOp::Lt, false, MinOrMaxFusion::ToMax),
            (vc::CompareOp::Le, false, MinOrMaxFusion::ToMax),
        ];

        for (compare_op, same_order, expected) in table {
            let scope = std::vec![
                receive(53),
                vector_load(55),
                compare(57, 53, 55, compare_op),
            ];
            let (lhs, rhs) = if same_order {
                (Val(53), Val(55))
            } else {
                (Val(55), Val(53))
            };

            assert_eq!(
                fuse_compare_and_select_into_min_or_max(pred(57), lhs, rhs, &scope),
                expected,
                "{compare_op:?} with the arms {}",
                if same_order { "in order" } else { "swapped" }
            );
        }
    }

    /// 🎯 `fcmp_select.mlir` — A CONDITION THAT IS NOT A COMPARE DOES NOT FUSE.
    ///
    /// The selection's cond is `%62`, an `agen.vector_load` of the state register
    /// (`fcmp_select.mlir:148,156`), so the `dyn_cast<ElementWiseCompareOp>` fails, both bools stay
    /// false, and `:64` emits the `sentient.vector_ternary … ternaryOp = select` instead.
    #[test]
    fn a_condition_loaded_from_memory_is_not_a_fusable_compare() {
        let scope = std::vec![vector_load(62), vector_load(64), vector_load(65)];

        assert_eq!(
            fuse_compare_and_select_into_min_or_max(pred(62), Val(64), Val(65), &scope),
            MinOrMaxFusion::NotFused
        );
    }

    /// 🎯 ARMS THAT ARE NOT THE COMPARE'S OWN OPERANDS DO NOT FUSE — neither order matches.
    ///
    /// ⭐ AND THE MISMATCH IS STRUCTURAL, NOT NOMINAL: the selection's `rhs` here is a *receive*
    /// where the compare's `op2` is a *load*, which is the case `skeleton` rejects on the op's name
    /// before any operand is looked at.
    #[test]
    fn arms_from_a_different_computation_do_not_fuse() {
        let scope = std::vec![
            receive(53),
            vector_load(55),
            receive(70),
            compare(57, 53, 55, vc::CompareOp::Gt),
        ];

        assert_eq!(
            fuse_compare_and_select_into_min_or_max(pred(57), Val(53), Val(70), &scope),
            MinOrMaxFusion::NotFused
        );
    }

    /// 🎯 AN ARM WITH NO DEFINING OP DOES NOT FUSE — `if (!lhs_def || …) return;`.
    ///
    /// `Val(200)` is a region argument as far as `scope` can tell, which is exactly the null
    /// `getDefiningOp()` the reference guards against.
    #[test]
    fn an_arm_that_is_a_region_argument_does_not_fuse() {
        let scope = std::vec![
            receive(53),
            vector_load(55),
            compare(57, 53, 200, vc::CompareOp::Gt),
        ];

        assert_eq!(
            fuse_compare_and_select_into_min_or_max(pred(57), Val(53), Val(200), &scope),
            MinOrMaxFusion::NotFused
        );
    }

    /// An operand off `port` whose data id has already been assigned.
    fn assigned(port: sen::Port, data_id: i32) -> sen::Operand {
        sen::Operand {
            data_id: Some(data_id),
            ..sen::Operand::from(port)
        }
    }

    /// `sentient.vector_mac` with all three data ids assigned.
    fn mac(a: i32, b: i32, c: i32) -> SenOp {
        SenOp::Sentient(sen::Op::VectorMac {
            mask: None,
            xrf_write_ptr: None,
            xrf_read_ptr: None,
            results: Vec::new(),
            op_a: assigned(sen::Port::Lx, a),
            op_b: assigned(sen::Port::Lrf(sen::LrfIndex::L1), b),
            op_c: assigned(sen::Port::Xrf, c),
            result: sen::ResultPorts::default(),
            mode: sen::FmaMode::FusedMulAdd,
            compute_precision: sen::Precision::Fp16,
            fold_mode: None,
            unroll_factor: sen::UnrollFactor::X1,
            xrf_read_incr: 0,
            xrf_write_incr: 0,
            data_transfer_only: false,
            dbg_name: None,
        })
    }

    /// `sentient.vector_binary` with both data ids assigned — `fmax.mlir:42` carries `1` and `2`.
    fn binary(a: i32, b: i32) -> SenOp {
        SenOp::Sentient(sen::Op::VectorBinary {
            mask: Val(0),
            op_a: assigned(sen::Port::Lx, a),
            op_b: assigned(sen::Port::Lrf(sen::LrfIndex::L1), b),
            binary_op: sen::Binary::Plain(sen::BinaryOp::Max),
            result: sen::ResultPorts::default(),
            compute_precision: sen::Precision::Fp16,
            fold_mode: None,
            unroll_factor: sen::UnrollFactor::X1,
            dbg_name: None,
        })
    }

    /// A unit whose body is `body`.
    fn sentient_unit(body: Vec<SenOp>) -> SentientProgramUnit<Target> {
        SentientProgramUnit {
            on: crate::islands::dataflow_ir::Units::one(crate::units::DfirUnit::Sfp, Val(0)),
            precision: None,
            body,
            arch: core::marker::PhantomData,
        }
    }

    /// Every data id the unit's computes carry, in walk order.
    fn data_ids(unit: &SentientProgramUnit<Target>) -> Vec<Option<i32>> {
        fn collect(ops: &[SenOp], into: &mut Vec<Option<i32>>) {
            for op in ops {
                match op {
                    SenOp::Sentient(sen::Op::VectorMac {
                        op_a, op_b, op_c, ..
                    })
                    | SenOp::Sentient(sen::Op::VectorTernary {
                        op_a, op_b, op_c, ..
                    }) => into.extend([op_a.data_id, op_b.data_id, op_c.data_id]),
                    SenOp::Sentient(sen::Op::VectorBinary { op_a, op_b, .. }) => {
                        into.extend([op_a.data_id, op_b.data_id]);
                    }
                    SenOp::Sentient(sen::Op::VectorUnary { op_a, .. }) => {
                        into.push(op_a.data_id);
                    }
                    SenOp::Sentient(sen::Op::For { body, .. }) => collect(body, into),
                    SenOp::Sentient(sen::Op::If {
                        then_body,
                        else_body,
                        ..
                    }) => {
                        collect(then_body, into);
                        collect(else_body, into);
                    }
                    _ => {}
                }
            }
        }
        let mut ids = Vec::new();
        collect(&unit.body, &mut ids);
        ids
    }

    /// 🎯 EVERY MAC AND BINARY DATA ID BECOMES UNASSIGNED, AT EVERY LOOP AND BRANCH DEPTH.
    ///
    /// `-1` is the `.td`'s "unassigned", i.e. [`sen::Operand::data_id`]` == None`, and the walk is
    /// `WalkOrder::PreOrder` over the whole unit — so a compute inside a `sentient.for` inside a
    /// `sentient.if` is reset like one at the top.
    #[test]
    fn resetting_the_fmas_unassigns_every_mac_and_binary_data_id() {
        let mut unit = sentient_unit(std::vec![
            mac(1, 2, 3),
            SenOp::Sentient(sen::Op::For {
                // ⭐ THE INDUCTION VARIABLE IS SYNTAX, NOT AN OPERAND — see [`sen::Op::For::iv`]. The
                // walk under test ignores it; it is here because the op always names one.
                iv: Val(11),
                bound: Val(1),
                carried: Vec::new(),
                dbg_name: None,
                body: std::vec![
                    binary(4, 5),
                    SenOp::Sentient(sen::Op::If {
                        predicate: sen::CmpPredicate::Eq,
                        lhs: Val(2),
                        rhs: Val(3),
                        yielded: Vec::new(),
                        dbg_name: None,
                        then_body: std::vec![mac(6, 7, 8)],
                        else_body: std::vec![binary(9, 10)],
                    }),
                ],
            }),
        ]);

        assert_eq!(
            data_ids(&unit),
            std::vec![
                Some(1),
                Some(2),
                Some(3),
                Some(4),
                Some(5),
                Some(6),
                Some(7),
                Some(8),
                Some(9),
                Some(10),
            ],
            "the fixture starts with every id assigned"
        );

        reset_sentient_fmas_if_exists(&mut unit);

        assert_eq!(data_ids(&unit), std::vec![None; 10]);
    }

    /// 🎯 A UNARY AND A TERNARY KEEP THEIRS — the reference's `else if` chain never reaches them.
    ///
    /// ⛔ `sentient::MacOp` and `sentient::BinaryOp` are the only two classes the `dyn_cast`s name
    /// (`VectorChainHelper.cpp:471-478`); `sentient.vector_unary` and `sentient.vector_ternary` are
    /// their own op classes and are left exactly as emitted.
    #[test]
    fn a_unary_and_a_ternary_keep_their_data_ids() {
        let mut unit = sentient_unit(std::vec![
            SenOp::Sentient(sen::Op::VectorUnary {
                mask: Val(0),
                op_a: assigned(sen::Port::Lx, 11),
                unary_op: sen::UnaryOp::Floor,
                result: sen::ResultPorts::default(),
                compute_precision: sen::Precision::Fp16,
                fold_mode: None,
                unroll_factor: sen::UnrollFactor::X1,
                dbg_name: None,
            }),
            SenOp::Sentient(sen::Op::VectorTernary {
                mask: None,
                op_a: assigned(sen::Port::IState(sen::IStateIndex::S0), 2),
                op_b: assigned(sen::Port::Lrf(sen::LrfIndex::L1), 3),
                op_c: assigned(sen::Port::Lx, 4),
                ternary_op: sen::TernaryOp::Select,
                result: sen::ResultPorts::default(),
                compute_precision: sen::Precision::Fp16,
                fold_mode: None,
                unroll_factor: sen::UnrollFactor::X1,
                unroll_incr_logical_result: false,
                dbg_name: None,
            }),
        ]);

        reset_sentient_fmas_if_exists(&mut unit);

        assert_eq!(
            data_ids(&unit),
            std::vec![Some(11), Some(2), Some(3), Some(4)],
            "fcmp_select.mlir:64 emits opADataID = 2, opBDataID = 3, opCDataID = 4"
        );
    }

    /// A one-unit program whose preamble is `preamble` and whose only body is `body`.
    fn dfir_program(preamble: Vec<DfirOp>, body: Vec<DfirOp>) -> DfirProgram<Target> {
        DfirProgram {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            grid: Grid::single(),
            preamble,
            units: ProgramUnits::of(
                DfirProgramUnit {
                    on: crate::islands::dataflow_ir::Units::one(crate::units::DfirUnit::Pe, Val(0)),
                    precision: None,
                    body,
                    arch: core::marker::PhantomData,
                },
                Vec::new(),
            ),
            arch: core::marker::PhantomData,
        }
    }

    /// `affine.for` over `body`, so a use can sit two region depths down.
    fn affine_for(iv: u32, body: Vec<DfirOp>) -> DfirOp {
        DfirOp::Affine(dfir_op::affine::Op::For {
            iv: Val(iv),
            lo: dfir_op::affine::Bound::Const(0),
            hi: dfir_op::affine::Bound::Const(10),
            carried: Vec::new(),
            body,
            dbg_name: None,
        })
    }

    /// `vectorchain.multiply_and_accumulate %a, %b, %acc[%mask]` — `const-vector-multiple-uses.mlir:241`.
    fn mac_reading(a: u32, b: u32, result: u32) -> DfirOp {
        DfirOp::VectorChain(vc::Op::MultiplyAccumulate {
            result: Val(result),
            a: Val(a),
            b: Val(b),
            acc: Val(300),
            reduction_map: AffineMap::identity(1),
            operand_ty: f16x64(),
            ty: f16x64(),
        })
    }

    /// 🎯 `const-vector-multiple-uses.mlir` — EVERY USE OF A VECTOR CONSTANT GETS ITS OWN CLONE, AND
    /// THE ORIGINAL IS GONE.
    ///
    /// The vendor case defines `%cst_1 = arith.constant dense<1.000000e+00> : vector<64xf16>` at the
    /// top of the function and reads it six times from four loop depths (`:185` and `:241,254,278,300,310,321`).
    /// Its expectation's function head (`:15-24`) keeps `arith.constant true`, `false` and the index
    /// constants and has **no** `dense<…>` left at all — every one moved next to its reader.
    ///
    /// ⭐ THE CLONE SITS IMMEDIATELY BEFORE ITS USER, at the user's own depth: that is what
    /// `OpBuilder builder(owner)` means, and it is what lets the next pass attach per-use facts to it.
    #[test]
    fn every_use_of_a_vector_constant_gets_its_own_clone() {
        let mut values = Values::default();
        for _ in 0..400 {
            let _ = values.mint();
        }
        let mut program = dfir_program(
            std::vec![dense_zero(1)],
            std::vec![
                mac_reading(10, 1, 20),
                affine_for(30, std::vec![mac_reading(11, 1, 21)]),
            ],
        );

        redefine_constant_vectors(&mut program, &mut values);

        assert!(
            program.preamble.is_empty(),
            "the original is erased once every use has a clone, leaving {:?}",
            program.preamble
        );

        let body = &program.units.iter().next().expect("one unit").body;
        let (top_clone, top_mac) = match &body[..] {
            [clone, top, _] => (clone, top),
            other => panic!("expected a clone ahead of the top-level mac, got {other:?}"),
        };
        assert!(
            matches!(
                top_clone,
                DfirOp::Arith(dfir_op::arith::Op::DenseConstant { .. })
            ),
            "the clone precedes its user"
        );

        let inner = match &body[2] {
            DfirOp::Affine(dfir_op::affine::Op::For { body, .. }) => body,
            other => panic!("expected the loop, got {other:?}"),
        };
        assert_eq!(inner.len(), 2, "the nested use got a clone inside the loop");

        // Three distinct values now, where the input had one.
        let clones = std::vec![
            dfir_op::results(top_clone)[0],
            dfir_op::results(&inner[0])[0],
        ];
        assert_ne!(clones[0], clones[1], "one clone per use, not one per value");
        assert_eq!(dfir_op::operands(top_mac)[1], clones[0]);
        assert_eq!(dfir_op::operands(&inner[1])[1], clones[1]);
        assert!(
            dfir_op::uses(Val(1), &program.preamble).is_empty()
                && dfir_op::uses(Val(1), body).is_empty(),
            "nothing reads the original any more"
        );
    }

    /// 🎯 ONE OP READING THE SAME CONSTANT TWICE GETS **TWO** CLONES.
    ///
    /// ⛔⛔ THIS IS WHY THE REWRITE IS PER OPERAND SLOT AND NOT A VALUE SUBSTITUTION.
    /// `const-vector-multiple-uses.mlir:310` is `multiply_and_accumulate %cst_0, %cst_1, %28` — two
    /// constants on one op — and the reference clones per **use**, so a substitution that mapped the
    /// value once would re-create exactly the sharing this function exists to undo.
    #[test]
    fn one_op_reading_a_constant_twice_gets_two_clones() {
        let mut values = Values::default();
        for _ in 0..400 {
            let _ = values.mint();
        }
        let mut program = dfir_program(std::vec![dense_zero(1)], std::vec![mac_reading(1, 1, 20)]);

        redefine_constant_vectors(&mut program, &mut values);

        let body = &program.units.iter().next().expect("one unit").body;
        assert_eq!(body.len(), 3, "two clones and the mac");
        let first = dfir_op::results(&body[0])[0];
        let second = dfir_op::results(&body[1])[0];
        assert_ne!(first, second);
        assert_eq!(
            dfir_op::operands(&body[2]),
            std::vec![first, second, Val(300)]
        );
    }

    /// 🎯 AN UNUSED VECTOR CONSTANT AND EVERY SCALAR CONSTANT ARE LEFT EXACTLY WHERE THEY WERE.
    ///
    /// `!use_empty()` is a real guard, not an optimisation. And `isa<VectorType, CustomVectorType>`
    /// excludes the scalars, which is why the vendor expectation still opens with `arith.constant 10`,
    /// `arith.constant true` and `arith.constant false` (`const-vector-multiple-uses.mlir:15-24`).
    #[test]
    fn an_unused_vector_constant_and_the_scalars_are_untouched() {
        let mut values = Values::default();
        let preamble = std::vec![
            dense_zero(1),
            DfirOp::Arith(dfir_op::arith::Op::Constant {
                result: Val(2),
                value: 10,
            }),
        ];
        let mut program = dfir_program(
            preamble.clone(),
            std::vec![DfirOp::Arith(dfir_op::arith::Op::AddI(
                dfir_op::arith::IntBinary {
                    result: Val(3),
                    lhs: Val(2),
                    rhs: Val(2),
                    ty: crate::islands::dataflow_ir::ty::ScalarTy::Index,
                }
            ))],
        );

        redefine_constant_vectors(&mut program, &mut values);

        assert_eq!(program.preamble, preamble);
        assert_eq!(values.issued(), 0, "no clone was minted");
    }

    /// 🎯 ALL TWELVE `vectorchain` BINARY OPERATORS MAP, AND THE MAP IS NOT AN IDENTITY.
    ///
    /// ⭐ THE TWELVE ARE THE WHOLE DOMAIN, which is why this port carries no `todo!` where the
    /// reference has a `DT_ERROR`.
    #[test]
    fn every_vectorchain_binary_operator_has_a_sentient_twin() {
        let table = std::vec![
            (vc::BinaryOp::And, sen::BinaryOp::And),
            (vc::BinaryOp::Or, sen::BinaryOp::Or),
            (vc::BinaryOp::Xnor, sen::BinaryOp::Xnor),
            (vc::BinaryOp::AndNot, sen::BinaryOp::AndNot),
            (vc::BinaryOp::Min, sen::BinaryOp::Min),
            (vc::BinaryOp::Max, sen::BinaryOp::Max),
            (vc::BinaryOp::AbsMin, sen::BinaryOp::AbsMin),
            (vc::BinaryOp::AbsMax, sen::BinaryOp::AbsMax),
            (vc::BinaryOp::Add, sen::BinaryOp::Add),
            (vc::BinaryOp::Mul, sen::BinaryOp::Mul),
            (vc::BinaryOp::MulDiv2, sen::BinaryOp::MulDiv2),
            (vc::BinaryOp::Sub, sen::BinaryOp::Sub),
        ];

        assert_eq!(table.len(), 12, "the whole of vc::BinaryOp");
        for (vb, expected) in table {
            assert_eq!(vector_binary_to_sentient_binary(vb), expected, "{vb:?}");
        }
    }

    /// 🎯 THE FOUR COMPARISONS THE ISA HAS BECOME THEIR `fcmp` — `gt` AND `ge` ARE NOT AMONG THEM.
    ///
    /// `fcmp_select.mlir` shows why: its input compare is `compare_gt` (`:136`) and its expectation is
    /// `binaryOp = #sentient<binary_operator fcmp_lt>` with the operands swapped (`:48`) — the caller
    /// reorders rather than asking for a `gt` that does not exist
    /// (`VectorChainToSentientPESFP.cpp:450-485`). The spellings are pinned because the attribute text
    /// is what a `CHECK` line matches.
    #[test]
    fn the_four_comparisons_the_isa_has_map_to_their_fcmp() {
        let table = std::vec![
            (vc::CompareOp::Eq, sen::BinaryOp::CompareEq, "fcmp_eq"),
            (vc::CompareOp::Neq, sen::BinaryOp::CompareNeq, "fcmp_neq"),
            (vc::CompareOp::Lt, sen::BinaryOp::CompareLt, "fcmp_lt"),
            (vc::CompareOp::Le, sen::BinaryOp::CompareLe, "fcmp_le"),
        ];

        for (vb, expected, spelling) in table {
            let got = vector_element_wise_compare_operator_to_sentient_binary_operator(vb);
            assert_eq!(got, expected, "{vb:?}");
            assert_eq!(got.spelling(), spelling);
        }
    }

    /// 🎯 THE SELECTION IS THE ONLY TERNARY — `fcmp_select.mlir:156` in, `:64`'s
    /// `ternaryOp = #sentient<ternary_operator select>` out.
    #[test]
    fn a_selection_is_the_only_ternary() {
        let selection = vc::Op::ElementWiseSelection {
            result: Val(67),
            cond: pred(62),
            lhs: Val(64),
            rhs: Val(65),
            mask: Some(pred(66)),
            ty: f16x64(),
        };

        assert_eq!(
            vector_ternary_to_sentient_ternary(&selection),
            sen::TernaryOp::Select
        );
        assert_eq!(sen::TernaryOp::Select.spelling(), "select");
    }

    // ═══════════════════════════════════════ 163 ═══════════════════════════════════════

    /// `affine_set<(d0) : (d0 - lb >= 0, -d0 + ub >= 0)>` — the ONE shape every `mask_set` in the
    /// authority tree's 825 `.mlir` files is written in.
    fn lane_span(lb: i64, ub: i64) -> IntegerSet {
        IntegerSet {
            dims: 1,
            symbols: 0,
            constraints: vec![
                Constraint {
                    expr: AffineExpr::dim(0).plus(AffineExpr::Const(-lb)),
                    is_equality: false,
                },
                Constraint {
                    expr: AffineExpr::dim(0).times(-1).plus(AffineExpr::Const(ub)),
                    is_equality: false,
                },
            ],
        }
    }

    /// `vectorchain.create_affine_mask %param {mask_set = #set} : ty` — the written-out form.
    fn mask_set_op(mask_set: IntegerSet, mask_parameter: Option<Val>, ty: Vector) -> vc::Op {
        vc::Op::CreateAffineMaskSet {
            result: Val(20),
            mask_set,
            mask_parameter,
            ty,
        }
    }

    /// The mask value, as the emitted `sentient.scalar_constant` would carry it.
    fn non_pt_mask(op: &vc::Op) -> Option<u32> {
        get_mask_value_constant_for_non_pt::<Target>(op).map(SliceMask::bits)
    }

    /// ⭐ 163/384 — THE ALL-LANES-LIVE MASK MASKS NO SLICE, over two widths and both island forms.
    ///
    /// `#set = affine_set<(d0) : (d0 - 64 >= 0, -d0 + 63 >= 0)>` on a `vector<64xi1>`
    /// (`dcc/test/Conversion/VectorChainToSentientPESFP/fnms_with_cast.mlir:20`, `:30`) lowers to
    /// `sentient.scalar_constant {value = 0 : si64}` (`:11`), and `fold_mode_df.mlir:92` writes THE
    /// SAME SET over a `vector<128xi1>` for the same answer (`:26`, `:29`) — one is `Lb 64 > Ub 63`
    /// at eight lanes a slice, the other at sixteen, and an empty range is zero either way.
    #[test]
    fn a_live_mask_masks_no_slice() {
        let sixty_four = mask_set_op(lane_span(64, 63), None, mask_ty());
        assert_eq!(non_pt_mask(&sixty_four), Some(0));

        let one_twenty_eight = mask_set_op(lane_span(64, 63), None, vec(128, ElemType::Int(1)));
        assert_eq!(non_pt_mask(&one_twenty_eight), Some(0));

        // The prefix form states the same set — [`vc::LaneMask::as_set`].
        let prefix = vc::Op::CreateAffineMask {
            result: Val(20),
            mask: vc::LaneMask::prefix_of(64, mask_ty()),
        };
        assert_eq!(non_pt_mask(&prefix), Some(0));
    }

    /// ⭐ 163/384 — THE MASK PARAMETER IS IGNORED HERE, WHICH IS THE ASYMMETRY WITH THE PT.
    ///
    /// `%19 = vectorchain.create_affine_mask %c0 {mask_set = #set2} : vector<128xi1>`
    /// (`fold_mode_df.mlir:92`) carries a parameter AND is lowered by the PE/SFP path, whose
    /// `CHECK-SENT-IR` is the same `{value = 0 : si64}` — entry 089 would fold `%c0` into the set's
    /// symbol first, and this side never looks at it.
    #[test]
    fn the_mask_parameter_is_not_read() {
        let with_parameter =
            mask_set_op(lane_span(64, 63), Some(Val(9)), vec(128, ElemType::Int(1)));
        assert_eq!(non_pt_mask(&with_parameter), Some(0));
    }

    /// 🎯 DERIVED — ONE BIT PER MASKED SLICE, and the PT's answer for the same mask is a COUNT.
    ///
    /// ⚠️ NO VENDOR CASE: every non-PT `mask_set` in the authority tree is the all-lanes-live shape
    /// above, so these come from the arithmetic at `VectorChainHelper.cpp:130-131`. Sixteen live
    /// lanes of 64 leaves slices 2..=7 masked (`0b1111_1100`); a mask that turns off the whole stick
    /// is `0b1111_1111`; and 48 live of 64 — the set IBM writes as `#set2` on the PT, where the
    /// answer is *two columns* (`dynamic_pt_masking.mlir:86`) — is slices 6 and 7 here.
    #[test]
    fn each_masked_slice_is_one_bit() {
        assert_eq!(
            non_pt_mask(&mask_set_op(lane_span(16, 63), None, mask_ty())),
            Some(0b1111_1100)
        );
        assert_eq!(
            non_pt_mask(&mask_set_op(lane_span(0, 63), None, mask_ty())),
            Some(0b1111_1111)
        );
        assert_eq!(
            non_pt_mask(&mask_set_op(lane_span(48, 63), None, mask_ty())),
            Some(0b1100_0000)
        );
        // The prefix form of that last one, through [`vc::LaneMask::as_set`].
        let prefix = vc::Op::CreateAffineMask {
            result: Val(20),
            mask: vc::LaneMask::prefix_of(48, mask_ty()),
        };
        assert_eq!(non_pt_mask(&prefix), Some(0b1100_0000));
    }

    /// 🎯 DERIVED — A NEGATIVE FIRST SLICE IS THE REFERENCE'S UNSIGNED WRAP, whose observable result
    /// is an empty loop: `from_slice - 1` under `unsigned` becomes huge and `i <= to_slice` is false
    /// at once (`VectorChainHelper.cpp:130-131`).
    #[test]
    fn a_negative_first_slice_masks_nothing() {
        assert_eq!(
            non_pt_mask(&mask_set_op(lane_span(-8, 63), None, mask_ty())),
            Some(0)
        );
    }

    /// 🎯 THE FOUR REFUSALS AND THE ONE STOP THIS PORT ADDS.
    ///
    /// ⚠️ DERIVED — the reference reaches each of these with an `emitOpError`, and no fixture does.
    #[test]
    fn a_mask_this_lowering_cannot_read_is_declined() {
        // "Mask affine set has to have one dimension." — two dims.
        let two_dims = IntegerSet {
            dims: 2,
            symbols: 0,
            constraints: lane_span(64, 63).constraints,
        };
        assert_eq!(non_pt_mask(&mask_set_op(two_dims, None, mask_ty())), None);

        // "Mask affine set should not have any symbols" — ⛔ WHERE A DYNAMIC MASK STOPS on this side.
        let symbolic = IntegerSet {
            dims: 1,
            symbols: 1,
            constraints: lane_span(64, 63).constraints,
        };
        assert_eq!(non_pt_mask(&mask_set_op(symbolic, None, mask_ty())), None);

        // "Mask affine set has to have constant bounds." — one-sided, so entry 063 answers false.
        let one_sided = IntegerSet {
            dims: 1,
            symbols: 0,
            constraints: vec![Constraint {
                expr: AffineExpr::dim(0).plus(AffineExpr::Const(-64)),
                is_equality: false,
            }],
        };
        assert!(!has_constant_bounds(&one_sided));
        assert_eq!(non_pt_mask(&mask_set_op(one_sided, None, mask_ty())), None);

        // `DT_CHECK(dim / sys_def.numSlicesPerStick > 0)` — fewer lanes than a stick has slices.
        assert_eq!(
            non_pt_mask(&mask_set_op(
                lane_span(0, 3),
                None,
                vec(4, ElemType::Int(1))
            )),
            None
        );

        // ⛔ THE STOP THIS PORT ADDS — `1 << to_slice` past the eight-bit slice mask. Twelve lanes
        // give one lane a slice, so the last slice is 11 and the reference shifts out of range.
        assert_eq!(
            non_pt_mask(&mask_set_op(
                lane_span(0, 11),
                None,
                vec(12, ElemType::Int(1))
            )),
            None
        );

        // Anything but the two mask forms has no `mask_set` at all.
        assert_eq!(
            non_pt_mask(&vc::Op::Rotate {
                result: Val(20),
                input: Val(19),
                position: Val(18),
                right_shift: true,
                input_ty: mask_ty(),
                ty: mask_ty(),
            }),
            None
        );
    }

    // ═══════════════════════════════════════ 164-165 ═══════════════════════════════════════

    /// THE THIRTY-FOUR PACKS OF `dcc/test/SFP/merge_and_pack.mlir:161-266`, each with the
    /// `binary_operator` its `CHECK-SENT-IR` names — in file order, which is the answer key for
    /// BOTH entries: every one has to pass entry 164 and then name its instruction through entry 165.
    ///
    /// ⛔ THE 24TH ROW IS THE POINT. The file's `%pack16` is written with `pack0`'s own indices and
    /// the reference answers `pack0`, because the scan returns the first match and the table has
    /// `pack0` eight rows earlier.
    ///
    /// One row per line is the whole point of a key, so `rustfmt` is held off this one.
    #[rustfmt::skip]
    fn merge_and_pack_key() -> Vec<(MergeOrPack, Vec<i32>, u32, bool)> {
        let merge = |width: sen::MergeWidth, high: bool| MergeOrPack::Merge { width, high };
        let pack = MergeOrPack::Pack;
        vec![
            (merge(sen::MergeWidth::W8, true), vec![1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31], 8, false),
            (merge(sen::MergeWidth::W8, false), vec![0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30], 8, false),
            (pack(sen::PackIndex::P25), vec![0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30], 8, false),
            (pack(sen::PackIndex::P26), vec![0, 2, 4, 6, 16, 18, 20, 22, 8, 10, 12, 14, 24, 26, 28, 30], 8, false),
            (pack(sen::PackIndex::P27), vec![0, 2, 16, 18, 4, 6, 20, 22, 8, 10, 24, 26, 12, 14, 28, 30], 8, false),
            (pack(sen::PackIndex::P12), vec![0, -1, 1, -1, 2, -1, 3, -1, 4, -1, 5, -1, 6, -1, 7, -1], 8, false),
            (pack(sen::PackIndex::P13), vec![8, -1, 9, -1, 10, -1, 11, -1, 12, -1, 13, -1, 14, -1, 15, -1], 8, false),
            (pack(sen::PackIndex::P14), vec![0, -1, 1, -1, 2, -1, 3, -1, 4, -1, 5, -1, 6, -1, 7, -1], 8, true),
            (pack(sen::PackIndex::P15), vec![8, -1, 9, -1, 10, -1, 11, -1, 12, -1, 13, -1, 14, -1, 15, -1], 8, true),
            (merge(sen::MergeWidth::W16, true), vec![1, 9, 3, 11, 5, 13, 7, 15], 16, false),
            (merge(sen::MergeWidth::W16, false), vec![0, 8, 2, 10, 4, 12, 6, 14], 16, false),
            (merge(sen::MergeWidth::W32, true), vec![2, 3, 10, 11, 6, 7, 14, 15], 16, false),
            (merge(sen::MergeWidth::W32, false), vec![0, 1, 8, 9, 4, 5, 12, 13], 16, false),
            (merge(sen::MergeWidth::W64, true), vec![4, 5, 6, 7, 12, 13, 14, 15], 16, false),
            (merge(sen::MergeWidth::W64, false), vec![0, 1, 2, 3, 8, 9, 10, 11], 16, false),
            (pack(sen::PackIndex::P0), vec![0, 1, 2, 3, 4, 5, 6, 7], 16, false),
            (pack(sen::PackIndex::P1), vec![15, 1, 2, 3, 4, 5, 6, 7], 16, false),
            (pack(sen::PackIndex::P2), vec![14, 15, 2, 3, 4, 5, 6, 7], 16, false),
            (pack(sen::PackIndex::P3), vec![13, 14, 15, 3, 4, 5, 6, 7], 16, false),
            (pack(sen::PackIndex::P4), vec![12, 13, 14, 15, 4, 5, 6, 7], 16, false),
            (pack(sen::PackIndex::P5), vec![11, 12, 13, 14, 15, 5, 6, 7], 16, false),
            (pack(sen::PackIndex::P6), vec![10, 11, 12, 13, 14, 15, 6, 7], 16, false),
            (pack(sen::PackIndex::P7), vec![9, 10, 11, 12, 13, 14, 15, 7], 16, false),
            // `%pack16` in, `pack0` out — `merge_and_pack.mlir:232` against the 24th
            // `binary_operator` of its `CHECK-SENT-IR`.
            (pack(sen::PackIndex::P0), vec![0, 1, 2, 3, 4, 5, 6, 7], 16, false),
            (pack(sen::PackIndex::P17), vec![8, 0, 1, 2, 3, 4, 5, 6], 16, false),
            (pack(sen::PackIndex::P18), vec![8, 9, 0, 1, 2, 3, 4, 5], 16, false),
            (pack(sen::PackIndex::P19), vec![8, 9, 10, 0, 1, 2, 3, 4], 16, false),
            (pack(sen::PackIndex::P20), vec![8, 9, 10, 11, 0, 1, 2, 3], 16, false),
            (pack(sen::PackIndex::P21), vec![8, 9, 10, 11, 12, 0, 1, 2], 16, false),
            (pack(sen::PackIndex::P22), vec![8, 9, 10, 11, 12, 13, 0, 1], 16, false),
            (pack(sen::PackIndex::P23), vec![8, 9, 10, 11, 12, 13, 14, 0], 16, false),
            (pack(sen::PackIndex::P8), vec![0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], 4, false),
            (pack(sen::PackIndex::P9), vec![0, 32, -1, -1, 4, 36, -1, -1, 8, 40, -1, -1, 12, 44, -1, -1, 16, 48, -1, -1, 20, 52, -1, -1, 24, 56, -1, -1, 28, 60, -1, -1], 4, false),
            (pack(sen::PackIndex::P24), vec![0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], 2, false),
        ]
    }

    /// `vectorchain.pack %x, %z {indices, repetition = 8 : index, sign_extend} : t, t, t` — every
    /// fixture of `merge_and_pack.mlir` is a full stick of `bits`-wide elements.
    fn pack_op(indices: Vec<i32>, bits: u32, sign_extend: bool) -> vc::Op {
        let ty = vec(1024 / u64::from(bits), ElemType::Int(bits));
        vc::Op::Pack {
            result: Val(50),
            op1: Val(48),
            op2: Val(49),
            mask: None,
            indices,
            repetition: 8,
            sign_extend,
            operand_ty: ty,
            ty,
        }
    }

    /// ⭐ 164/384 + 165/384 — THE WHOLE ANSWER KEY: all thirty-four packs of
    /// `dcc/test/SFP/merge_and_pack.mlir` pass the validity check and name the instruction their
    /// `CHECK-SENT-IR` says, `%pack16 → pack0` included.
    #[test]
    fn every_vendor_pack_names_its_instruction() {
        for (expected, indices, bits, sign_extend) in merge_and_pack_key() {
            let op = pack_op(indices.clone(), bits, sign_extend);
            let witness = PackOrShuffle::of(&op).expect("a pack is one");
            let valid = check_validity_of_pack_and_shuffle_lowering(witness)
                .unwrap_or_else(|| panic!("{expected:?} {indices:?} was refused"));
            assert_eq!(valid.indices(), indices.as_slice());
            assert_eq!(valid.repetition(), PackRepetition::Eight);
            assert_eq!(
                merge_type_from_indices(&valid, sign_extend, Bits(bits)),
                Some(expected),
                "{indices:?} at {bits} bits"
            );
        }
    }

    /// ⭐ 165/384 — `sign_extend` IS THE WHOLE DIFFERENCE BETWEEN TWO PAIRS OF ROWS. `pack12` and
    /// `pack14` share their indices and their width (`merge_and_pack.mlir:176`, `:182`), so asking
    /// for the wrong one finds neither `pack12` nor anything else.
    #[test]
    fn the_sign_extending_twins_are_not_interchangeable() {
        let pack12 = vec![0, -1, 1, -1, 2, -1, 3, -1, 4, -1, 5, -1, 6, -1, 7, -1];
        let op = pack_op(pack12, 8, false);
        let valid = check_validity_of_pack_and_shuffle_lowering(PackOrShuffle::of(&op).unwrap())
            .expect("valid");

        assert_eq!(
            merge_type_from_indices(&valid, false, Bits(8)),
            Some(MergeOrPack::Pack(sen::PackIndex::P12))
        );
        assert_eq!(
            merge_type_from_indices(&valid, true, Bits(8)),
            Some(MergeOrPack::Pack(sen::PackIndex::P14))
        );
    }

    /// 🎯 THE FOUR REFUSALS OF ENTRY 164, and the one it deliberately lets through.
    ///
    /// ⚠️ DERIVED — no fixture is refused; each of these is an `emitError` in the reference.
    #[test]
    fn an_unlowerable_pack_is_refused() {
        let ok = vec![0, 1, 2, 3, 4, 5, 6, 7];

        // `!is_any_of(repetition, 4, 8)`.
        let mut two = pack_op(ok.clone(), 16, false);
        if let vc::Op::Pack { repetition, .. } = &mut two {
            *repetition = 2;
        }
        assert_eq!(
            check_validity_of_pack_and_shuffle_lowering(PackOrShuffle::of(&two).unwrap()),
            None
        );
        assert_eq!(PackRepetition::of(2), None);

        // "Indices should be integers >= -1".
        let minus_two = pack_op(vec![0, 1, 2, 3, 4, 5, 6, -2], 16, false);
        assert_eq!(
            check_validity_of_pack_and_shuffle_lowering(PackOrShuffle::of(&minus_two).unwrap()),
            None
        );

        // `repetition * indices.size() != vec_size` — eight indices, eight repetitions, but the
        // sixteen-bit stick holds sixty-four lanes and this one says it holds a hundred and twenty
        // eight.
        let mut wrong_width = pack_op(ok.clone(), 16, false);
        if let vc::Op::Pack { ty, .. } = &mut wrong_width {
            ty.len = 128;
        }
        assert_eq!(
            check_validity_of_pack_and_shuffle_lowering(PackOrShuffle::of(&wrong_width).unwrap()),
            None
        );

        // ⛔ `index > 2 * (int)indices.size()` IS STRICTLY GREATER, so sixteen passes for eight
        // indices and seventeen does not — reproduced, not tightened. Neither names an instruction.
        let at_the_bound = pack_op(vec![16, 1, 2, 3, 4, 5, 6, 7], 16, false);
        let admitted =
            check_validity_of_pack_and_shuffle_lowering(PackOrShuffle::of(&at_the_bound).unwrap())
                .expect("2 * size is admitted");
        assert_eq!(merge_type_from_indices(&admitted, false, Bits(16)), None);

        let past_it = pack_op(vec![17, 1, 2, 3, 4, 5, 6, 7], 16, false);
        assert_eq!(
            check_validity_of_pack_and_shuffle_lowering(PackOrShuffle::of(&past_it).unwrap()),
            None
        );

        // Neither a pack nor a shuffle.
        assert!(
            PackOrShuffle::of(&vc::Op::CreateAffineMask {
                result: Val(20),
                mask: vc::LaneMask::prefix_of(64, mask_ty()),
            })
            .is_none()
        );
    }

    /// ⭐ 164/384 + 165/384 — A REPETITION OF FOUR IS VALID AND NAMES NOTHING IN THIS TABLE. It is a
    /// GCVT/FCVT shuffle's number, taken by entry 277 instead
    /// (`VectorChainToSentientPESFP.cpp:633-641`).
    #[test]
    fn a_repetition_of_four_is_valid_and_matches_no_row() {
        let ty = f16x64();
        let shuffle = vc::Op::Shuffle {
            result: Val(44),
            input: Val(43),
            variable: Vec::new(),
            pad: Vec::new(),
            mask: None,
            indices: (0..16).collect(),
            repetition: 4,
            input_ty: ty,
            ty,
        };

        let valid =
            check_validity_of_pack_and_shuffle_lowering(PackOrShuffle::of(&shuffle).unwrap())
                .expect("four is one of the two accepted repetitions");
        assert_eq!(valid.repetition(), PackRepetition::Four);
        assert_eq!(merge_type_from_indices(&valid, false, Bits(16)), None);
        assert!(
            merge_and_pack_insts()
                .iter()
                .all(|inst| inst.repetition() == 8)
        );
    }

    /// ⭐ 165/384 — THE TABLE IS THIRTY-FOUR ROWS AND ITS ORDER IS THE ANSWER. `pack16` sits eight
    /// rows behind an identical `pack0` (`VectorChainHelper.cpp:374`, `:383`), so nothing can reach
    /// it — which is why the vendor's own `%pack16` lowers to `pack0`.
    #[test]
    fn the_table_is_ordered_and_pack16_is_unreachable() {
        let insts = merge_and_pack_insts();
        assert_eq!(insts.len(), 34);

        let pack0 = &insts[18];
        let pack16 = &insts[26];
        assert_eq!(pack0.name(), MergeOrPack::Pack(sen::PackIndex::P0));
        assert_eq!(pack16.name(), MergeOrPack::Pack(sen::PackIndex::P16));
        assert_eq!(pack0.vec(), pack16.vec());
        assert_eq!(pack0.element_bit_width(), pack16.element_bit_width());
        assert_eq!(pack0.sign_extend(), pack16.sign_extend());

        // `pack24`'s checksum counts its forty-eight pads: 960 - 48.
        assert_eq!(insts[0].name(), MergeOrPack::Pack(sen::PackIndex::P24));
        assert_eq!(insts[0].sum(), 912);
        assert_eq!(insts[0].size(), 64);
    }

    /// 🎯 DERIVED — A NARROWER ELEMENT REACHES THE SAME ROW `scale` LANES AT A TIME.
    ///
    /// ⚠️ NO VENDOR CASE: every fixture states its permutation at the row's own width. `merge32l`'s
    /// `{0, 1, 8, 9, 4, 5, 12, 13}` is written for sixteen-bit elements, and the inner loop at
    /// `VectorChainHelper.cpp:402-408` makes the same instruction name the eight-bit permutation that
    /// moves those halves in pairs. ⛔ AND A ROW WITH A PAD CANNOT WIDEN — `pack12` at scale two
    /// would demand an index of `-2`, which entry 164 refuses outright.
    #[test]
    fn a_row_widens_by_its_scale() {
        let paired = vec![0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27];
        let op = pack_op(paired, 8, false);
        let valid =
            check_validity_of_pack_and_shuffle_lowering(PackOrShuffle::of(&op).unwrap()).unwrap();

        assert_eq!(
            merge_type_from_indices(&valid, false, Bits(8)),
            Some(MergeOrPack::Merge {
                width: sen::MergeWidth::W32,
                high: false
            })
        );

        // The same list one lane out names nothing at all.
        let wrong = pack_op(
            vec![0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 28],
            8,
            false,
        );
        let valid_wrong =
            check_validity_of_pack_and_shuffle_lowering(PackOrShuffle::of(&wrong).unwrap())
                .unwrap();
        assert_eq!(merge_type_from_indices(&valid_wrong, false, Bits(8)), None);
    }

    /// 🎯 A ZERO ELEMENT WIDTH IS THE ONE STOP THIS PORT ADDS TO ENTRY 165 — `scale` divides by it
    /// (`VectorChainHelper.cpp:396`).
    #[test]
    fn a_zero_element_width_names_nothing() {
        let op = pack_op(vec![0, 1, 2, 3, 4, 5, 6, 7], 16, false);
        let valid =
            check_validity_of_pack_and_shuffle_lowering(PackOrShuffle::of(&op).unwrap()).unwrap();
        assert_eq!(merge_type_from_indices(&valid, false, Bits(0)), None);
    }

    // ═══════════════════════════════════════ 228 ═══════════════════════════════════════

    /// ⭐ 228/384 — ONE COUNTER WALKS THE WHOLE UNIT, AND AN ID ALREADY THERE IS LEFT ALONE.
    ///
    /// `reset_sentient_fmas_if_exists` puts every operand back to `-1` (entry 156), so the mac takes
    /// 0, 1, 2 in A-B-C order and the binary continues at 3 — except its A, which keeps the 41 it
    /// came in with, exactly as the reference's `== -1` guard says.
    #[test]
    fn the_missing_data_ids_are_numbered_from_the_reuse_count() {
        let mut unit = sentient_unit(vec![mac(1, 2, 3), binary(4, 5)]);
        reset_sentient_fmas_if_exists(&mut unit);
        if let SenOp::Sentient(sen::Op::VectorBinary { op_a, .. }) = &mut unit.body[1] {
            op_a.data_id = Some(41);
        }

        validate_lowering_and_set_missing_parameters(&mut unit, &OperandReuse::default());

        assert_eq!(
            data_ids(&unit),
            vec![Some(0), Some(1), Some(2), Some(41), Some(3)]
        );
    }

    // ═══════════════════════════════════════ 229 ═══════════════════════════════════════

    /// ⭐ 229/384 — THE MASK CONSTANT AS THE REFERENCE PRINTS IT: `sentient.scalar_constant
    /// {value = 0 : si64} : index` (`fnms_with_cast.mlir:11`) for entry 163's all-lanes-live set.
    #[test]
    fn a_non_pt_mask_becomes_an_index_constant() {
        let mut values = Values::default();
        let mask = get_mask_value_for_non_pt::<Target>(
            &mask_set_op(lane_span(64, 63), None, mask_ty()),
            &mut values,
        );

        assert_eq!(
            mask,
            Some(NonPtMask {
                op: SenOp::Sentient(sen::Op::ScalarConstant {
                    value: 0,
                    result: Val(0),
                    reg_locale: sen::RegType::Imm,
                    ty: ScalarTy::Index,
                    is_symbol: false,
                }),
                value: Val(0),
            })
        );
    }

    /// 🎯 230 + 231 — THE ROUND TRIP, AND THE ONE SPELLING THAT DOES NOT SURVIVE IT.
    ///
    /// The pair is written as inverses (`VectorChainHelper.hpp:196-243`) and is not one: entry 230
    /// accepts `mxint4` and entry 231 has no arm for the type it produces.
    #[test]
    fn every_spelling_but_mxint4_round_trips_through_its_element_type() {
        for spelling in [
            TypeSpelling::Fp16,
            TypeSpelling::Bf16,
            TypeSpelling::Fp32,
            TypeSpelling::F8E4M3Fn,
            TypeSpelling::F8E5M2,
            TypeSpelling::Mxfp4,
            TypeSpelling::Mxfp8,
            TypeSpelling::I16,
            TypeSpelling::I8,
            TypeSpelling::I4,
        ] {
            assert_eq!(
                convert_type_to_string(convert_string_to_type(spelling)),
                Some(spelling)
            );
        }

        // ⛔ `mxint4` IN, `llvm_unreachable("unknown type")` BACK.
        assert_eq!(
            convert_string_to_type(TypeSpelling::Mxint4),
            ElemType::MxInt(4)
        );
        assert_eq!(convert_type_to_string(ElemType::MxInt(4)), None);

        // The four spellings entry 277's own table is written in (`VectorChainHelper.cpp:245-267`).
        assert_eq!(
            convert_type_to_string(ElemType::F32),
            Some(TypeSpelling::Fp32)
        );
        assert_eq!(
            convert_type_to_string(ElemType::F16),
            Some(TypeSpelling::Fp16)
        );
        assert_eq!(
            convert_type_to_string(ElemType::Bf16),
            Some(TypeSpelling::Bf16)
        );
        assert_eq!(
            convert_type_to_string(ElemType::F8E5M2),
            Some(TypeSpelling::F8E5M2)
        );
    }

    /// The vendor's own GCVT table, `dcc/test/Conversion/VectorChainToSentientPESFP/gcvt_sen1p5.mlir`
    /// — its `imm = 0` pack (two casts), its `imm = 1` shuffle (one cast) and its `imm = 16`
    /// shuffle, which is the repetition-four row.
    #[test]
    fn the_vendor_gcvt_table_names_its_own_three_shapes() {
        let f8 = |len| Vector {
            len,
            elem: ElemType::F8E5M2,
        };
        let f16 = |len| Vector {
            len,
            elem: ElemType::F16,
        };
        let bf16 = |len| Vector {
            len,
            elem: ElemType::Bf16,
        };

        // `%9 = vectorchain.cast %7 : vector<128xf16>, vector<128xf8E5M2>` — twice, then
        // `%11 = vectorchain.pack %9, %10 {indices = [0..15], repetition = 8}`.
        let to_f8 = CastSource::of(&vc::Op::Cast {
            result: Val(9),
            input: Val(7),
            input_ty: f16(128),
            ty: f8(128),
        })
        .unwrap();
        let pack = vc::Op::Pack {
            result: Val(11),
            op1: Val(9),
            op2: Val(10),
            mask: None,
            indices: (0..16).collect(),
            repetition: 8,
            sign_extend: false,
            operand_ty: f8(128),
            ty: f8(128),
        };
        let checked =
            check_validity_of_pack_and_shuffle_lowering(PackOrShuffle::of(&pack).unwrap()).unwrap();
        assert_eq!(
            gcvt_or_fcvt_type_from_indices_and_cast_inputs(&checked, &[to_f8, to_f8]),
            Some(CvtInst::Gcvt2(sen::BinaryGcvt::Imm0))
        );
        // ⛔ THE SAME PERMUTATION WITH ONE SOURCE IS NOT `gcvt_imm0` — arity is part of the row.
        assert_eq!(
            gcvt_or_fcvt_type_from_indices_and_cast_inputs(&checked, &[to_f8]),
            None
        );

        // `%28 = vectorchain.cast %27 : vector<128xf8E5M2>, vector<128xf16>` then
        // `%29 = vectorchain.shuffle input(%28) {indices = [0..7], repetition = 8}`.
        let from_f8 = CastSource::of(&vc::Op::Cast {
            result: Val(28),
            input: Val(27),
            input_ty: f8(128),
            ty: f16(128),
        })
        .unwrap();
        let shuffle = vc::Op::Shuffle {
            result: Val(29),
            input: Val(28),
            variable: Vec::new(),
            pad: Vec::new(),
            mask: None,
            indices: (0..8).collect(),
            repetition: 8,
            input_ty: f16(128),
            ty: f16(64),
        };
        let checked =
            check_validity_of_pack_and_shuffle_lowering(PackOrShuffle::of(&shuffle).unwrap())
                .unwrap();
        assert_eq!(
            gcvt_or_fcvt_type_from_indices_and_cast_inputs(&checked, &[from_f8]),
            Some(CvtInst::Gcvt1(sen::UnaryGcvt::Imm1))
        );

        // `%43 = vectorchain.cast %42 : vector<64xf16>, vector<64xbf16>` then
        // `%44 = vectorchain.shuffle input(%43) {indices = [0..15], repetition = 4}`.
        let to_bf16 = CastSource::of(&vc::Op::Cast {
            result: Val(43),
            input: Val(42),
            input_ty: f16(64),
            ty: bf16(64),
        })
        .unwrap();
        let shuffle = vc::Op::Shuffle {
            result: Val(44),
            input: Val(43),
            variable: Vec::new(),
            pad: Vec::new(),
            mask: None,
            indices: (0..16).collect(),
            repetition: 4,
            input_ty: bf16(64),
            ty: bf16(64),
        };
        let checked =
            check_validity_of_pack_and_shuffle_lowering(PackOrShuffle::of(&shuffle).unwrap())
                .unwrap();
        assert_eq!(
            gcvt_or_fcvt_type_from_indices_and_cast_inputs(&checked, &[to_bf16]),
            Some(CvtInst::Gcvt1(sen::UnaryGcvt::Imm16))
        );
    }

    /// 🎯 340/384 — THE SPLAT COMES FIRST AND THE CAST CONTRIBUTES NOTHING.
    ///
    /// An SFP operand off a receive that folded a `vectorchain.select` in from below is read by a
    /// `vectorchain.cast` and by a `dataflow.send` to the LXSU. The cast is one of the five classes
    /// that fold INTO the operand, so the only ports forwarded are the splat's `east` and the
    /// send's `lx` — in that order, because the splat is pushed ahead of the walk.
    #[test]
    fn an_operand_forwards_its_splat_and_its_send_but_not_its_cast() {
        let v = vec(64, ElemType::F16);
        let (to, from) = Link::<Sfp, Lxsu>::between(Val(20), Val(21)).ends();
        let scope = vec![
            DfirOp::Dataflow(dfir_op::dataflow::Op::Receive {
                result: Val(1),
                from,
                ty: v,
            }),
            DfirOp::VectorChain(vc::Op::Cast {
                result: Val(2),
                input: Val(1),
                input_ty: v,
                ty: v,
            }),
            DfirOp::Dataflow(dfir_op::dataflow::Op::Send {
                to,
                data: Val(1),
                ty: v,
                dir: None,
            }),
            DfirOp::Dataflow(dfir_op::dataflow::Op::GetUnit {
                result: Val(21),
                residency: Residency::Global,
                unit: DfirUnit::Lxsu,
                num_folds: None,
            }),
        ];
        let mut operand = VectorOperand {
            kind: VectorOperandType::Link,
            op: OpId::at(&[0]),
            values: Vec::new(),
            orig_precision: None,
            on_the_fly_conv_precision: None,
            splat: Some(sen::Port::East),
        };

        let mut ports = Vec::new();
        analyze_and_fill_operand_forwarding::<Target>(
            ComputeComp::Sfp,
            Some(&operand),
            &scope,
            &mut ports,
        );
        assert_eq!(ports, vec![sen::Port::East, sen::Port::Lx]);

        // ⛔ AND AN `nfwd` OPERAND RETURNS BEFORE THE SPLAT, so neither port is reached — the early
        // return is above the push, not below it.
        operand.kind = VectorOperandType::Nfwd;
        let mut refused = Vec::new();
        analyze_and_fill_operand_forwarding::<Target>(
            ComputeComp::Sfp,
            Some(&operand),
            &scope,
            &mut refused,
        );
        assert_eq!(refused, Vec::new());
    }

    /// 🎯 341/384 — A HOISTED CONSTANT'S THREE READERS SPLIT THREE WAYS.
    ///
    /// `arith.constant dense<1>` sits OUTSIDE the program unit, as the reference's own comment says
    /// the canonicalizer leaves it. Inside the unit a `dataflow.send` reads it — the op under
    /// analysis, so it is pushed once and then skipped as itself — beside a `vectorchain.neg` that is
    /// neither send nor store, and outside it a `vectorchain.fast_exp` that is not the unit's at all.
    #[test]
    fn a_hoisted_constants_readers_split_into_merged_unfused_and_dangling() {
        let v = vec(64, ElemType::F16);
        let (to, _) = Link::<Sfp, Lxsu>::between(Val(20), Val(21)).ends();
        let scope = vec![
            DfirOp::Arith(dfir_op::arith::Op::DenseConstant {
                result: Val(0),
                splat: 1,
                ty: v,
            }),
            DfirOp::Dataflow(dfir_op::dataflow::Op::ProgramUnit {
                units: vec![Val(30)],
                precision: None,
                body: vec![
                    DfirOp::Dataflow(dfir_op::dataflow::Op::Send {
                        to,
                        data: Val(0),
                        ty: v,
                        dir: None,
                    }),
                    DfirOp::VectorChain(vc::Op::Neg {
                        result: Val(5),
                        input: Val(0),
                        mask: None,
                        input_ty: v,
                        ty: v,
                    }),
                ],
            }),
            DfirOp::VectorChain(vc::Op::FastExp {
                result: Val(9),
                input: Val(0),
                input_ty: v,
                ty: v,
            }),
            DfirOp::Dataflow(dfir_op::dataflow::Op::GetUnit {
                result: Val(21),
                residency: Residency::Global,
                unit: DfirUnit::Lxsu,
                num_folds: None,
            }),
        ];
        let from = VectorOperand {
            kind: VectorOperandType::Constant,
            op: OpId::at(&[0]),
            values: Vec::new(),
            orig_precision: None,
            on_the_fly_conv_precision: None,
            splat: None,
        };

        let mut is_visited = BTreeMap::new();
        let mut analysis = FusionAnalysis::default();
        analyze_non_compute_ops_for_fusion::<Target>(
            &OpId::at(&[1]),
            &OpId::at(&[1, 0]),
            &from,
            ComputeComp::Sfp,
            &mut is_visited,
            &scope,
            &mut analysis,
        );

        assert_eq!(analysis.to_operands.len(), 1, "the send, pushed on entry");
        assert_eq!(
            analysis.to_operands[0]
                .as_ref()
                .and_then(VectorOperand::name),
            Some(sen::Port::Lx)
        );
        // ⛔ THE `neg` FORWARDS THE CONSTANT'S OWN OPERAND, which is neither a send nor a store, so
        // the from-op may not be erased.
        assert!(!analysis.is_fusion_respected);
        // ⛔ AND THE READER OUTSIDE THE UNIT IS DANGLING, which only the constant arm can see.
        assert!(analysis.is_dangling_ops_present_after_fusion);
        // Nothing merged, so nothing was marked visited — the send was skipped as the current op.
        assert!(is_visited.is_empty());
    }

    /// 🎯 342/384 — AN `sfpring` RESULT FORWARDS THE RING **AND** EMITS `sentient.set_send_dst`.
    ///
    /// The SFP's result is read by a `vectorchain.cast` whose single user sends to another SFP, and
    /// on this ISA that destination is the ring. The forwarded port comes from the cast, while the
    /// `set_send_dst` names the unit the SEND carries — which is why the cast has to be stepped over.
    #[test]
    fn an_sfpring_result_forwards_the_ring_and_names_the_sends_unit() {
        let v = vec(64, ElemType::F16);
        let (to, _) = Link::<Sfp, Sfp>::between(Val(20), Val(21)).ends();
        let scope = vec![
            DfirOp::VectorChain(vc::Op::FastExp {
                result: Val(1),
                input: Val(0),
                input_ty: v,
                ty: v,
            }),
            DfirOp::VectorChain(vc::Op::Cast {
                result: Val(2),
                input: Val(1),
                input_ty: v,
                ty: v,
            }),
            DfirOp::Dataflow(dfir_op::dataflow::Op::Send {
                to,
                data: Val(2),
                ty: v,
                dir: None,
            }),
            DfirOp::Dataflow(dfir_op::dataflow::Op::GetUnit {
                result: Val(21),
                residency: Residency::Global,
                unit: DfirUnit::Sfp,
                num_folds: None,
            }),
        ];

        let mut to_operands = Vec::new();
        let mut result_forwarding = Vec::new();
        let mut logical_result_forwarding = None;
        let mut emitted = Vec::new();
        analyze_and_fill_result_forwarding::<Target>(
            &OpId::at(&[0]),
            ComputeComp::Sfp,
            &scope,
            &mut to_operands,
            &mut result_forwarding,
            &mut logical_result_forwarding,
            &mut emitted,
        );

        assert_eq!(result_forwarding, vec![sen::Port::SfpRing]);
        // ⛔ THE RING IS NOT `istate`, so it forwards as a result and not as the logical one.
        assert_eq!(logical_result_forwarding, None);
        assert_eq!(
            emitted,
            vec![SenOp::Sentient(sen::Op::SetSendDst { units: to })]
        );
        assert_eq!(to_operands.len(), 1);
    }
}
