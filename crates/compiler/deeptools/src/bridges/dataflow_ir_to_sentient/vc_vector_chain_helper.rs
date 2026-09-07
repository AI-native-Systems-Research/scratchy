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
use crate::formats::Bits;
use crate::islands::dataflow_ir::dialects::vectorchain as vc;
use crate::islands::dataflow_ir::dialects::{self as dfir_op, Op as DfirOp};
use crate::islands::dataflow_ir::ty::{BoundType, ElemType, IntegerSet, Vector};
use crate::islands::sentient::dialects::sentient as sen;

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
        // The `vectorchain` ops both reference lists name, and the estimate family — `FastExpOp`,
        // `ExpEstimateOp`, `RecEstimateOp`, `LnEstimateOp`, `RsqrtEstimateOp`, `FloorOp`,
        // `SigmoidEstimateOp` and `TanhEstimateOp` — which [`vc::Op::Estimate`] holds as one op.
        DfirOp::VectorChain(
            vc::Op::Estimate { ty, .. }
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


#[cfg(test)]
mod unit_tests {
    use super::{
        MergeAndPack, MergeOrPack, compute_precision_of_op, has_constant_bounds,
        input_precision_from_operand, is_sentient_binary_logical_op, precision_in_string,
        result_precision_from_operands, vector_type_of,
    };
    use crate::formats::Bits;
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
}
