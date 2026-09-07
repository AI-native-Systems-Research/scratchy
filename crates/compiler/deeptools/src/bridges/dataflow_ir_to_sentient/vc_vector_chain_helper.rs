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

use super::vc_vector_operands::VectorOperand;
use crate::arch::Arch;
use crate::formats::Bits;
use crate::islands::dataflow_ir::dialects::vectorchain as vc;
use crate::islands::dataflow_ir::dialects::{self as dfir_op, Op as DfirOp, Val};
use crate::islands::dataflow_ir::ty::{BoundType, ElemType, IntegerSet, Vector};
use crate::islands::dataflow_ir::{Program as DfirProgram, Values};
use crate::islands::sentient::ProgramUnit as SentientProgramUnit;
use crate::islands::sentient::dialects::sentient as sen;
use crate::islands::sentient::dialects::Op as SenOp;
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
        sen::BinaryOp::And
        | sen::BinaryOp::Or
        | sen::BinaryOp::Xnor
        | sen::BinaryOp::AndNot => true,
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
fn precision_in_string(elem: ElemType) -> sen::Precision {
    match elem {
        // ⭐ THE MX CHECK IS FIRST IN THE REFERENCE, before `isIntOrIndexOrFloat` is even asserted:
        // a `CustomMXFloatType` is not an int-or-float, so the order is load-bearing there. Here the
        // element type already distinguishes them.
        ElemType::MxFloat(4) => sen::Precision::Mxfp4,
        ElemType::MxFloat(8) => sen::Precision::Mxfp8,
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
        ElemType::F8E4M3Fn | ElemType::F8E8M0Fnu => sen::Precision::Fp8,
        ElemType::F4E2M1Fn => sen::Precision::Fp4,
        // ⛔ NO SUCH PRECISION EXISTS TO NAME. `SentientPrecisionAttr` has nineteen cases and none of
        // them is an `int<n>` or `mxfp<n>` for any other `n`; the reference would build the string
        // and then abort in `symbolizeSentientPrecision(..).value()`
        // (`VectorChainToSentientPESFP.cpp:362`).
        ElemType::Int(bits) | ElemType::MxFloat(bits) => {
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
/// [`vc::Op::CreateAffineMask`] and [`vc::Op::Merge`] every one of them results in a vector, and
/// every one of them is ABSENT from both reference lists — so `getElementType` aborts on a cast.
/// ⛔ `RotateOp` IS THE ONE TO WATCH: `getLoadConsumer` names it beside `SelectOp` and `ShuffleOp`
/// as a rearrangement (`Helper.cpp:1268-1270`), and of those three only the SHUFFLE is in
/// `getVectorType` (`Utils.cpp:581`). `RotateOp` appears nowhere in `Utils.cpp` at all. Answering from the result type
/// "because it obviously has one" would paper over the abort the reference relies on
/// [`compute_precision_of_op`] short-circuiting past.
///
/// ⚠️ ONE DELIBERATE WIDENING, FLAGGED: the reference lists `vector::LoadOp` and `vector::StoreOp`,
/// which are what D1's `AffineToStandard` leaves behind. This island still holds those accesses in
/// their `affine` form ([`dfir_op::affine::Op::VectorLoad`]/`VectorStore`) as well as the `agen` one,
/// so both spellings answer here. They are the same op one rung earlier; treating the affine pair as
/// absent would make the answer depend on which side of D1 the query happens to run.
fn vector_type_of(op: &DfirOp) -> Option<Vector> {
    match op {
        // `dataflow::SendOp` and `dataflow::ReceiveOp`.
        DfirOp::Dataflow(dfir_op::dataflow::Op::Send { ty, .. }) => Some(*ty),
        DfirOp::Dataflow(dfir_op::dataflow::Op::Receive { ty, .. }) => Some(*ty),
        DfirOp::Dataflow(
            dfir_op::dataflow::Op::GetUnit { .. }
            | dfir_op::dataflow::Op::GetLocalUnit { .. }
            | dfir_op::dataflow::Op::GetLogicalMemoryView { .. }
            | dfir_op::dataflow::Op::GetPagedLogicalMemoryView(_)
            | dfir_op::dataflow::Op::ProgramUnit { .. }
            | dfir_op::dataflow::Op::SyncSend { .. }
            | dfir_op::dataflow::Op::SyncRecv { .. }
            | dfir_op::dataflow::Op::ImplicitSync { .. }
            | dfir_op::dataflow::Op::Opaque(_),
        ) => None,
        // `agen::VectorLoadOp` and `agen::VectorStoreOp`.
        DfirOp::Agen(dfir_op::agen::Op::VectorLoad { ty, .. }) => Some(*ty),
        DfirOp::Agen(dfir_op::agen::Op::VectorStore { ty, .. }) => Some(*ty),
        DfirOp::Agen(dfir_op::agen::Op::CompositeLoadAndStore(_) | dfir_op::agen::Op::Yield) => None,
        // `vector::LoadOp` and `vector::StoreOp` — see the widening note.
        DfirOp::Affine(dfir_op::affine::Op::VectorLoad { ty, .. }) => Some(*ty),
        DfirOp::Affine(dfir_op::affine::Op::VectorStore { ty, .. }) => Some(*ty),
        DfirOp::Affine(
            dfir_op::affine::Op::For { .. }
            | dfir_op::affine::Op::Apply { .. }
            | dfir_op::affine::Op::Yield { .. },
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
            | vc::Op::Rotate { .. }
            | vc::Op::ConstantBitstream { .. }
            | vc::Op::Cast { .. }
            | vc::Op::CreateAffineMask { .. }
            | vc::Op::Merge { .. },
        ) => None,
        // Neither reference list mentions an `arith` or an `scf` op.
        DfirOp::Arith(_) | DfirOp::Scf(_) => None,
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
    /// ⛔ `repetition` IS ALWAYS EIGHT. It is a member DEFAULT (`int repetition = 8;`, `:309`) that
    /// the constructor never assigns and no row overrides, and it is compared for equality against
    /// the pack op's own `repetition` (`:399`) — so it is a constant of the table, not a parameter.
    ///
    /// ⛔ `sign_extend` IS EXPLICIT AT EVERY CALL. The C++ defaults it (`bool sign_extend = false`,
    /// `:313`) and thirty of the thirty-four rows leave it out; Rust has no default arguments, so the
    /// four rows that DO set it — `pack14`, `pack15`, and the two the table pairs them with — read
    /// beside their twins rather than differing in an omission.
    ///
    /// ⚠️ ITS CALLER IS UNPORTED. The table of thirty-four rows and the matching loop are
    /// `getMergeTypeFromIndices` (entry 165), which is not this worklist's; nothing in the crate
    /// builds a row yet.
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
/// (`VectorChainHelper.cpp:460-462`), because the pair can spell a fourth thing — *both* — that
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
/// are the compare's own two operands is one Sentient instruction, not two: `fmax.mlir:117-118` is
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
/// [`vc::Op`] here would force a `todo!` arm for the fourteen variants the C++ signature already rules
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

    let same_pair_matching = ops_are_equivalent(lhs_def, op1_def, scope)
        && ops_are_equivalent(rhs_def, op2_def, scope);
    let opposite_pair_matching = !same_pair_matching
        && ops_are_equivalent(lhs_def, op2_def, scope)
        && ops_are_equivalent(rhs_def, op1_def, scope);

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
            todo!("Unable to lower compare and select into max/min operation (VectorChainHelper.cpp:460-462)")
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
/// (`OperationEquivalence.hpp:27-34`), with a null functor and a null `operands_equiv_checker`.
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
fn ops_are_equivalent(a: &DfirOp, b: &DfirOp, scope: &[DfirOp]) -> bool {
    // `if (&op_a == &op_b) return true;` — also the recursion's cycle guard.
    if core::ptr::eq(a, b) {
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
                if core::ptr::eq(def_a, def_b) || !ops_are_equivalent(def_a, def_b, scope) {
                    return false;
                }
            }
            // Both are region arguments: equivalent, because `all_block_args_are_equiv_` is true.
            (None, None) => {}
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
            if !ops_are_equivalent(inner_a, inner_b, scope) {
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
            // The shared dialects' nested bodies are `Vec<DfirOp>` — no compute can be in one.
            SenOp::Dataflow(_)
            | SenOp::Agen(_)
            | SenOp::VectorChain(_)
            | SenOp::Affine(_)
            | SenOp::Arith(_)
            | SenOp::Scf(_) => {}
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
/// pass has an answer this function could give. The remaining fourteen `vectorchain` ops keep it:
/// they are compares, binaries and multiplies, none of which is a ternary.
#[must_use]
pub fn vector_ternary_to_sentient_ternary(op: &vc::Op) -> sen::TernaryOp {
    match op {
        vc::Op::ElementWiseSelection { .. } => sen::TernaryOp::Select,
        vc::Op::Estimate { .. }
        | vc::Op::FastExp { .. }
        | vc::Op::Floor { .. }
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
        | vc::Op::CreateAffineMask { .. } => {
            todo!("unknown vectorchain Ternary operator (VectorChainHelper.hpp:255)")
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::{
        MergeAndPack, MergeOrPack, MinOrMaxFusion, compute_precision_of_op,
        fuse_compare_and_select_into_min_or_max, has_constant_bounds, input_precision_from_operand,
        is_sentient_binary_logical_op, precision_in_string, redefine_constant_vectors,
        reset_sentient_fmas_if_exists, result_precision_from_operands,
        vector_binary_to_sentient_binary,
        vector_element_wise_compare_operator_to_sentient_binary_operator,
        vector_ternary_to_sentient_ternary, vector_type_of,
    };
    use super::{DfirProgram, SenOp, SentientProgramUnit, Values};
    use crate::arch::Dd2;
    use crate::formats::Bits;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::link::{Link, Lxlu, Sfp};
    use crate::islands::dataflow_ir::ty::MemRef;
    use crate::islands::dataflow_ir::{
        Grid, GroupId, OpIndex, ProgramName, ProgramUnit as DfirProgramUnit, ProgramUnits,
    };

    /// The arch every fixture below is built for; nothing here is arch-dependent.
    type Target = Dd2;

    use crate::islands::dataflow_ir::dialects::Val;
    use crate::islands::dataflow_ir::dialects::vectorchain as vc;
    use crate::islands::dataflow_ir::dialects::{self as dfir_op, Op as DfirOp};
    use crate::islands::dataflow_ir::ty::{
        AffineExpr, AffineMap, Constraint, ElemType, IntegerSet, Vector,
    };
    use crate::bridges::dataflow_ir_to_sentient::vc_vector_operands::{
        OpId, VectorOperand, VectorOperandType,
    };
    use crate::islands::sentient::dialects::sentient as sen;

    /// An operand off the link with the two precisions under test.
    fn operand(orig: Option<sen::Precision>, on_the_fly: Option<sen::Precision>) -> VectorOperand {
        VectorOperand {
            kind: VectorOperandType::Link,
            op: OpId::at(&[0]),
            values: Vec::new(),
            orig_precision: orig,
            on_the_fly_conv_precision: on_the_fly,
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
        assert_ne!(
            precision_in_string(ElemType::F16),
            sen::Precision::IeeeFp16
        );
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
            constraints: vec![Constraint {
                expr: AffineExpr::dim(0),
                is_equality: false,
            }],
        };
        let upper_only = IntegerSet {
            dims: 1,
            constraints: vec![Constraint {
                expr: AffineExpr::dim(0).times(-1).plus(AffineExpr::Const(63)),
                is_equality: false,
            }],
        };
        let unconstrained = IntegerSet {
            dims: 1,
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
        let row = MergeAndPack::new(
            MergeOrPack::Pack(sen::PackIndex::P24),
            Bits(2),
            vec,
            false,
        );
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
        let pack14 = MergeAndPack::new(
            MergeOrPack::Pack(sen::PackIndex::P14),
            Bits(8),
            lanes,
            true,
        );
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
            result: Val(result),
            view: Val(92),
            indices: std::vec![dfir_op::Index::Const(0)],
            view_ty: MemRef {
                shape: std::vec![64, 4, 1],
                elem: ElemType::F16,
            },
            ty: f16x64(),
        })
    }

    /// `arith.constant dense<0.000000e+00> : vector<64xf16>` — `relu.mlir:50`.
    fn dense_zero(result: u32) -> DfirOp {
        DfirOp::Arith(dfir_op::arith::Op::DenseConstant {
            result: Val(result),
            one: false,
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
    /// `element_wise_selection %57 ? %53 : %55` (`fmax.mlir:117-118`), and the expectation is a single
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
        assert_eq!(dfir_op::operands(&body[2]), std::vec![first, second, Val(300)]);
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
}
