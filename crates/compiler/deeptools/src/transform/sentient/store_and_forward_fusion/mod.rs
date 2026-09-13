// SPDX-License-Identifier: Apache-2.0
//
// ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ CRUSTIFY SENTIENT-PASSES CAMPAIGN — READ THIS BEFORE YOU FILL AN ANCHOR IN THIS FILE.        ║
// ║ Campaign statement: crustify-senpass/TASK.md   ·   worklist: crustify-senpass/UNITS.tsv      ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
//
// 1. THE AUTHORITY IS THE C++ TREE, NOT THE EXTRACT.
//       /Users/nickm/git/deeptools-src/<file>:<line>     (deeptools|master|a0d29abbed — repo_info.txt)
//    Every citation below resolves against that revision. `crustify-senpass/cpp/sentient.cpp` says
//    WHICH functions are in scope and IN WHAT ORDER; its bodies were verified byte-identical to the
//    authority (656/656, 962,619 bytes, two negative controls), so either may be read — but the
//    authority file is the one that carries the surrounding declarations you will need.
//    ⛔ /Users/nickm/git/deeptools is a DIFFERENT revision. ⛔ The pod is not reachable from here.
//
// 2. THESE PASSES REWRITE SentientIR IN PLACE. They are NOT a conversion between rungs like bridges
//    1–4: input and output are both `src/islands/sentient/`. Expect to EXTEND that island — ops
//    gaining an assigned register, a pinned address, a rolled loop — not to emit into a new one.
//    WHY IT MATTERS: bridge 2's emission assigns NO registers (`index: None` in 39 of 41 sites),
//    faithfully, because the reference's SentientIR carries `regIndex = -1 : i32` in all 54
//    occurrences of the committed golden corpus. THESE passes turn -1 into a real register file and
//    index. Without them ProgIR gets -1 where an instruction needs a register and the backend
//    refuses with `Register initialization out of boundary` (observed on lxsu0:LRF0, l3lu:LBR2).
//
// 3. PORTED MEANS THE WHOLE FUNCTION INCLUDING ITS EFFECT. For an in-place pass the effect IS the
//    port: WHICH ops are rewritten, WHICH attributes are set to WHAT, and IN WHAT ORDER. A hand
//    attempt on bridge 2 extracted each function's decision rule into a documented predicate,
//    omitted the part that changed the IR, and reported it done — nothing called any of it.
//    Droppable: only the mechanism for REACHING operands (walking uses, memoising, positioning a
//    builder). ⛔ If the island cannot express a result, EXTEND THE ISLAND. Deciding a function is
//    unnecessary is NOT the porter's call, and a predicate is not a port.
//
// 4. HARD CAPS — MEASURED, AND THEY BIND. Bridge 2's first 144 functions cost 43 h wall clock and
//    586 M cache-read tokens; of 30,991 lines produced only 5,306 were implementation (12,333 doc
//    comments, 12,575 tests). Per ported function:
//      • 8 DOC LINES: the `/// Replaces: eNNN_name` anchor, one line of what it does, any TRAP.
//        No tutorials, no restating the C++, no design essays.
//      • ONE TEST. Two only where the vendor's own case AND a negative both apply.
//      • `cargo check -p deeptools` + `cargo test -p deeptools` ONCE PER BATCH, not per function.
//      • Do NOT re-verify citations — the review pass owns that.
//      • Do NOT grep the crate to discover types; the anchor names what you need.
//    NOT capped: correctness, and the emission.
//
// 5. THIS IS A PURE-LOGIC PORT WITH NO C ANYWHERE. Whatever generic C-to-Rust conventions say:
//    ❌ no bindgen/allowlist/-sys, ❌ no `ffi::`/`extern "C"`, ❌ no `CRUSTIFY_<FILE>` switch,
//    ❌ no `Foo`/`FooRef`/`FooMut` triple, ❌ no `unsafe`, ❌ no C-vs-Rust equivalence harness.
//
// 6. CRATE RULES BIND YOU — `crates/compiler/deeptools/CLAUDE.md`, read it in full.
//    🛑 NEVER RUNTIME REFUSE: no `Result`, no `Err(`, no `.ok_or`, no `assert!`, no `debug_assert!`.
//    A closed set is an `enum`; an invariant is a TYPE; newtypes, never raw scalars. No strings for
//    closed sets. `Arch`/`Model`/`Workload` flow through. `todo!` NAMING UNPORTED WORK IS ALLOWED
//    here — and ⛔ never substitute a stand-in op to dodge one.
//
// 7. ANCHORS: each `// crustify:todo: eNNN_name` below is one scheduled unit. Replace it with the
//    ported item carrying the doc anchor `/// Replaces: eNNN_name`. ⛔ NEVER DELETE AN ANCHOR YOU
//    DID NOT PORT — on bridge 2 a deleted anchor was indistinguishable from a finished unit and the
//    campaign reported DONE having silently lost 149 of 384 functions, the biggest in its span.
//
// 8. GATE: `cargo check -p deeptools` and `cargo test -p deeptools`. ⛔ NEVER run the workspace or
//    the acceptance build in an agent worktree — ~6 GB of target/ each, and this host is tight.
//
// 9. 43 OF THE 52 PASSES CONSUME AN ANALYSIS THAT IS OUT OF SCOPE (122 of the 656 units name one):
//    `Analyses/` (Liveness, PropagationAnalysis, GraphColoring, ExpressionEvaluatorUtils,
//    InstructionEstimation, TimeStamps, RegisterPressureAnalysis, AddressPinningScheme,
//    XRFRegisterAnalyzer, CorrelationAnalysis, RedundantDefinitionEliminationTree, …) and
//    `RegisterInitialization/` (Collector, Evaluator, Selector, Transformer, UniformGrouper) —
//    ~11,700 lines NOT in this campaign. Per pass: `crustify-senpass/OUTSIDE-DEPS.tsv`; per unit:
//    `OUTSIDE-UNITS.tsv`. ⭐ When a unit needs one, port the part that is present and
//    `todo!("<Analysis>::<method> — out of campaign scope")` for the part that is not, leaving the
//    anchor FILLED so the unit is not lost. ⛔ DO NOT INVENT THE ANALYSIS and do not substitute a
//    constant for its result. Extending `src/islands/sentient/` is a different case and IS expected.

//! `StoreAndForwardFusion.cpp` — 11 of the campaign's 656 units (dependency level(s) [0, 1, 2, 3]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e223_appendToVector` | 223 | 0 | 4 | `dcc/src/Transform/Sentient/StoreAndForwardFusion.cpp:94` |
//! | `e224_findFusibleOp` | 224 | 0 | 29 | `dcc/src/Transform/Sentient/StoreAndForwardFusion.cpp:117` |
//! | `e225_isOpTrivialMacOp` | 225 | 0 | 31 | `dcc/src/Transform/Sentient/StoreAndForwardFusion.cpp:150` |
//! | `e226_isOpTrivialBinaryOp` | 226 | 0 | 21 | `dcc/src/Transform/Sentient/StoreAndForwardFusion.cpp:183` |
//! | `e227_fillOperandForFusibleOp` | 227 | 0 | 27 | `dcc/src/Transform/Sentient/StoreAndForwardFusion.cpp:350` |
//! | `e228_updateFusibleOp` | 228 | 0 | 41 | `dcc/src/Transform/Sentient/StoreAndForwardFusion.cpp:387` |
//! | `e385_appendValueForwarding` | 385 | 1 | 6 | `dcc/src/Transform/Sentient/StoreAndForwardFusion.cpp:100` |
//! | `e386_isEligibleToFuse` | 386 | 1 | 87 | `dcc/src/Transform/Sentient/StoreAndForwardFusion.cpp:219` |
//! | `e387_fillOperandForTrivialOp` | 387 | 1 | 19 | `dcc/src/Transform/Sentient/StoreAndForwardFusion.cpp:318` |
//! | `e480_FuseStoreAndForward` | 480 | 2 | 119 | `dcc/src/Transform/Sentient/StoreAndForwardFusion.cpp:434` |
//! | `e541_runOnOperation` | 541 | 3 | 24 | `dcc/src/Transform/Sentient/StoreAndForwardFusion.cpp:555` |

// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET — [`run_on_operation`] (e541) is the entry, and
// there is no ported D29-D75 pass driver to call it, so nothing but this file's own tests reaches
// anything here. CI runs clippy with `-D warnings`.
// ⭐ REMOVE THIS WITH THAT DRIVER, not with an anchor: e541 is filled and the pass is still unwired.
#![allow(dead_code)]

use super::analyses::{InstructionEstimator, OutOfScopeInstructionEstimator};
use super::rematerialization_pass::InBlock;
use super::utils::fold_mode_attribute_if_exists;
use crate::arch::Arch;
use crate::bridges::dataflow_ir_to_sentient::tf_cfgs_dataflow_conditional_tree::{
    DbgNamePrefix, new_dbg_name_from_list,
};
use crate::islands::dataflow_ir::ty::GenericComp;
use crate::islands::sentient::Program;
use crate::islands::sentient::dialects::sentient::{Operand, Port, Precision, ResultPorts};
use crate::islands::sentient::dialects::{self as dialects, Op, dataflow, sentient};
use crate::model::Model;
use crate::workload::Workload;

/// WHETHER THE WALK IS STEPPING OVER `fold_AB` PAIRS — `is_fold_AB_mode`.
///
/// ⛔⛔ IT IS A STRIDE, NOT A FLAG. `fold_AB` splits one compute across two ADJACENT ops, a
/// `fold_AB_A` and the `fold_AB_B` after it (e480 recognises the pair at `:442-448`), so the candidate
/// before a pair is TWO ops back — the reference's `getPrevNode()->getPrevNode()` (`:120-121`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FoldAbMode {
    /// `is_fold_AB_mode = false` — one op per step, the default argument (`:84`).
    Off,
    /// `true` — the ops come in `(fold_AB_A, fold_AB_B)` pairs.
    On,
}

impl FoldAbMode {
    /// How far back one step of the walk goes.
    #[must_use]
    const fn stride(self) -> usize {
        match self {
            FoldAbMode::Off => 1,
            FoldAbMode::On => 2,
        }
    }
}

/// `llvm::dyn_cast<sentient::MacOp>` AS A WITNESS — the operand bundles a `sentient.vector_mac` carries.
///
/// ⛔ THE C++ OP-CLASS NAMES ARE KEPT HERE, AND THEY COLLIDE WITH THIS CRATE'S OPERATOR ENUMS:
/// [`MacOp`]/[`BinaryOp`]/[`UnaryOp`] in this module are the three OP classes the pass fuses, while
/// [`sentient::BinaryOp`] and [`sentient::UnaryOp`] are the OPERATOR attributes those ops carry. A
/// bare name below is always the op class; the operator is always spelled `sentient::`.
#[derive(Debug, Clone, Copy)]
pub struct MacOp<'a> {
    /// `getOpA()` and `getOpAForwarding()`.
    pub op_a: &'a Operand,
    /// `getOpB()`/`getOpBForwarding()`.
    pub op_b: &'a Operand,
    /// `getOpC()`/`getOpCForwarding()` — the accumulator.
    pub op_c: &'a Operand,
    /// `getResultForwarding()`, `getResultPrecision()` and `getUnrollIncrResult()`.
    pub result: &'a ResultPorts,
    /// `getUnrollFactorVal()` (`SentientOps.cpp:1678`), which e386 matches against the fusible op's.
    pub unroll_factor: sentient::UnrollFactor,
}

impl<'a> MacOp<'a> {
    /// `llvm::dyn_cast<sentient::MacOp>(op)`.
    #[must_use]
    pub fn of(op: &'a Op) -> Option<MacOp<'a>> {
        match op {
            Op::Sentient(sentient::Op::VectorMac {
                op_a,
                op_b,
                op_c,
                result,
                unroll_factor,
                ..
            }) => Some(MacOp {
                op_a,
                op_b,
                op_c,
                result,
                unroll_factor: *unroll_factor,
            }),
            _ => None,
        }
    }
}

/// `llvm::dyn_cast<sentient::BinaryOp>` AS A WITNESS — a `sentient.vector_binary`'s two operands, its
/// operator and its result bundle. See [`MacOp`] for the name collision.
#[derive(Debug, Clone, Copy)]
pub struct BinaryOp<'a> {
    /// `getOpA()`/`getOpAForwarding()`.
    pub op_a: &'a Operand,
    /// `getOpB()`/`getOpBForwarding()`.
    pub op_b: &'a Operand,
    /// `getBinaryOp()`, carried with its logical forward — see [`sentient::Binary`].
    pub op: sentient::Binary,
    /// `getResultForwarding()`, `getResultPrecision()` and `getUnrollIncrResult()`.
    pub result: &'a ResultPorts,
    /// `getUnrollFactorVal()` (`SentientOps.cpp:2214`).
    pub unroll_factor: sentient::UnrollFactor,
}

impl<'a> BinaryOp<'a> {
    /// `llvm::dyn_cast<sentient::BinaryOp>(op)`.
    #[must_use]
    pub fn of(op: &'a Op) -> Option<BinaryOp<'a>> {
        match op {
            Op::Sentient(sentient::Op::VectorBinary {
                op_a,
                op_b,
                binary_op,
                result,
                unroll_factor,
                ..
            }) => Some(BinaryOp {
                op_a,
                op_b,
                op: *binary_op,
                result,
                unroll_factor: *unroll_factor,
            }),
            _ => None,
        }
    }
}

/// `llvm::dyn_cast<sentient::UnaryOp>` AS A WITNESS — a `sentient.vector_unary`'s one operand, its
/// operator and its result bundle. See [`MacOp`] for the name collision.
#[derive(Debug, Clone, Copy)]
pub struct UnaryOp<'a> {
    /// `getOpA()`/`getOpAForwarding()`.
    pub op_a: &'a Operand,
    /// `getUnaryOp()` — what decides whether this op may be fused into at all; see [`is_reduction`].
    pub op: sentient::UnaryOp,
    /// `getResultForwarding()`, `getResultPrecision()` and `getUnrollIncrResult()`.
    pub result: &'a ResultPorts,
    /// `getUnrollFactorVal()` (`SentientOps.cpp:2307`).
    pub unroll_factor: sentient::UnrollFactor,
}

impl<'a> UnaryOp<'a> {
    /// `llvm::dyn_cast<sentient::UnaryOp>(op)`.
    #[must_use]
    pub fn of(op: &'a Op) -> Option<UnaryOp<'a>> {
        match op {
            Op::Sentient(sentient::Op::VectorUnary {
                op_a,
                unary_op,
                result,
                unroll_factor,
                ..
            }) => Some(UnaryOp {
                op_a,
                op: *unary_op,
                result,
                unroll_factor: *unroll_factor,
            }),
            _ => None,
        }
    }
}

/// Replaces: e223_appendToVector
///
/// Appends one forwarding list onto a running list of value forwardings (`:94-98`).
///
/// ⛔ TRAP: `if (!array_attr) return;` GUARDS THE **NULL** `ArrayAttr`, NOT AN EMPTY ONE — a
/// `vector_binary` has no `opCForwarding` at all, so e227 leaves that out-param default-constructed
/// and this appends nothing for it. [`Operand::forwarding`] is a `Vec<Port>` that is empty rather than
/// absent, and appending an empty slice is the same statement, so the guard becomes the slice itself.
pub fn append_to_vector(vector_attr: &mut Vec<Port>, array_attr: &[Port]) {
    vector_attr.extend_from_slice(array_attr);
}

/// `is_any_of(unary_op.getUnaryOp(), reduction_abs_max, reduction_abs_min, reduction_add,
/// reduction_max, reduction_min)` (`:129-132`) — the five operators that make a `vector_unary`
/// UNFUSIBLE.
///
/// ⛔ A REDUCTION FALLS **THROUGH** in [`find_fusible_op`] rather than stopping the walk (`:128-136`
/// has no `else`), so one already queued for deletion is still stepped over.
#[must_use]
const fn is_reduction(unary_op: sentient::UnaryOp) -> bool {
    matches!(
        unary_op,
        sentient::UnaryOp::ReductionAbsMax
            | sentient::UnaryOp::ReductionAbsMin
            | sentient::UnaryOp::ReductionAdd
            | sentient::UnaryOp::ReductionMax
            | sentient::UnaryOp::ReductionMin
    )
}

/// Replaces: e224_findFusibleOp
///
/// The op a trivial compute at `op` may fuse into: the nearest `vector_mac`, `vector_binary` or
/// non-reduction `vector_unary` before it in the same block, stepping over `sentient.scalar_constant`s,
/// `dataflow.get_unit`s and ops already queued for deletion (`:110-147`).
///
/// ⛔ [`None`] IS THE REFERENCE'S "HANDS BACK ITS OWN ARGUMENT" (`:146`), AND ONLY THE FIRST OF
/// e480'S TWO GUARDS CATCHES IT. `fusible_ops.first == op` (`:466`) fires; the second compares
/// `fusible_ops.second` against `op` (`:470`) after calling this on `op->getNextNode()`, so it
/// CANNOT fire — see [`fuse_store_and_forward`] for what the reference does with the failure it
/// misses, and why declining diverges deliberately.
/// ⛔ AND STEPPING OFF THE FRONT OF THE BLOCK IS A NULL DEREFERENCE THERE: in `fold_AB` mode
/// `op->getPrevNode()->getPrevNode()` (`:120-121`, `:140-141`) dereferences null for an op at index 0
/// or 1. This answers [`None`] — a deliberate divergence from a crash.
/// ⭐ `fusible_op->getBlock() == op->getBlock()` (`:123`) IS ALWAYS TRUE WHERE IT IS TESTED, because
/// `getPrevNode()` is a sibling by construction; the loop guard is the non-null test alone.
/// ⛔ `sentient::ConstantOp` IS `sentient.scalar_constant` ALONE (`SentientOps.td:848`) — a
/// `sentient.vector_constant` is a different op class and STOPS the walk.
#[must_use]
pub fn find_fusible_op(
    op: InBlock,
    block: &[Op],
    to_be_deleted: &[InBlock],
    is_fold_ab_mode: FoldAbMode,
) -> Option<InBlock> {
    let stride = is_fold_ab_mode.stride();
    let mut at = InBlock::at(op.index().checked_sub(stride)?);
    loop {
        let fusible_op = block.get(at.index())?;
        if matches!(
            fusible_op,
            Op::Sentient(sentient::Op::VectorMac { .. } | sentient::Op::VectorBinary { .. })
        ) {
            return Some(at);
        }
        if let Some(unary_op) = UnaryOp::of(fusible_op)
            && !is_reduction(unary_op.op)
        {
            return Some(at);
        }
        let steppable = matches!(
            fusible_op,
            Op::Sentient(sentient::Op::ScalarConstant { .. })
                | Op::Dataflow(dataflow::Op::GetUnit { .. })
        ) || to_be_deleted.contains(&at);
        if !steppable {
            return None;
        }
        at = InBlock::at(at.index().checked_sub(stride)?);
    }
}

/// Replaces: e225_isOpTrivialMacOp
///
/// Whether a `vector_mac` computes nothing but a forward: `0*B+C` or `A*0+C` forwarding `opC` or the
/// result, or `lrf<n>*1+0` forwarding that operand or the result (`:148-180`).
///
/// ⛔ `opC != zero` RETURNS EITHER WAY (`:159-166`): an accumulating MAC is trivial only in the
/// zero-operand form, so the two `lrf*1` arms are unreachable for one.
/// ⭐ `stringifySentientComputePort(port).contains("lrf")` IS `Port::Lrf(_)` — of the sixty-three
/// spellings (`SentientTypes.td:165-227`) only `lrf<n>` holds it, so the test becomes a match.
/// ⛔ AND EACH `lrf` ARM REQUIRES THE **OTHER** OPERAND'S FORWARDING TO BE EMPTY, not its own.
#[must_use]
pub fn is_op_trivial_mac_op(mac_op: MacOp<'_>) -> bool {
    if mac_op.op_c.port != Port::Zero {
        return (mac_op.op_a.port == Port::Zero || mac_op.op_b.port == Port::Zero)
            && (!mac_op.op_c.forwarding.is_empty() || !mac_op.result.forwarding.is_empty());
    }
    if matches!(mac_op.op_a.port, Port::Lrf(_))
        && mac_op.op_b.port == Port::One
        && (!mac_op.op_a.forwarding.is_empty() || !mac_op.result.forwarding.is_empty())
        && mac_op.op_b.forwarding.is_empty()
    {
        return true;
    }
    if matches!(mac_op.op_b.port, Port::Lrf(_))
        && mac_op.op_a.port == Port::One
        && (!mac_op.op_b.forwarding.is_empty() || !mac_op.result.forwarding.is_empty())
        && mac_op.op_a.forwarding.is_empty()
    {
        return true;
    }
    false
}

/// Replaces: e226_isOpTrivialBinaryOp
///
/// Whether a `vector_binary` computes nothing but a forward: `lrf<n> or 0` forwarding that operand or
/// the result (`:182-204`).
///
/// ⛔ `or0` AND NOTHING ELSE (`:185`) — the operator is spelled `or0` because *"and and or are
/// reserved keyword in c++"*, and `and0`, `add` or a compare is never trivial.
/// ⛔ THE ZERO OPERAND IS `zero`, NOT `one` AS IN [`is_op_trivial_mac_op`], and each arm again requires
/// the OTHER operand's forwarding to be empty.
/// ⭐ A LOGICAL-RESULT FORWARD CANNOT REACH HERE: [`sentient::ForwardingOp`] admits no `or0`, so
/// [`sentient::Binary::op`] is total and the `Forwarding` arm always answers a different operator.
#[must_use]
pub fn is_op_trivial_binary_op(binary_op: BinaryOp<'_>) -> bool {
    if binary_op.op.op() != sentient::BinaryOp::Or {
        return false;
    }
    if matches!(binary_op.op_a.port, Port::Lrf(_))
        && binary_op.op_b.port == Port::Zero
        && (!binary_op.op_a.forwarding.is_empty() || !binary_op.result.forwarding.is_empty())
        && binary_op.op_b.forwarding.is_empty()
    {
        return true;
    }
    if matches!(binary_op.op_b.port, Port::Lrf(_))
        && binary_op.op_a.port == Port::Zero
        && (!binary_op.op_b.forwarding.is_empty() || !binary_op.result.forwarding.is_empty())
        && binary_op.op_a.forwarding.is_empty()
    {
        return true;
    }
    false
}

/// THE FOUR FORWARDING LISTS AND THE RESULT PRECISION ONE COMPUTE OP CONTRIBUTES — the five
/// out-params `fillOperandForFusibleOp` (e227) and `fillOperandForTrivialOp` (e387) fill.
///
/// ⛔ AN ABSENT LIST IS AN EMPTY ONE: a `vector_binary` never touches `opC_forwarding` and a
/// `vector_unary` touches neither `opB` nor `opC`, leaving those `ArrayAttr`s null — which is what
/// [`append_to_vector`] appends as nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Forwardings {
    /// `opA_forwarding`.
    pub op_a: Vec<Port>,
    /// `opB_forwarding` — empty for a `vector_unary`.
    pub op_b: Vec<Port>,
    /// `opC_forwarding` — empty for anything but a `vector_mac`.
    pub op_c: Vec<Port>,
    /// `result_forwarding`.
    pub result: Vec<Port>,
    /// `result_precision`.
    pub result_precision: Precision,
}

impl Forwardings {
    /// Replaces: e227_fillOperandForFusibleOp
    ///
    /// The forwardings and result precision of the op a trivial compute is about to fuse into — a
    /// `vector_mac`, `vector_binary` or `vector_unary` (`:341-379`).
    ///
    /// ⛔ [`None`] IS THE `emitError` + `LogicalResult::failure()` ARM (`:373-377`): e480 abandons the
    /// fusion on it, and the diagnostic is the reference's only other effect.
    /// ⛔ THE THREE `dyn_cast`s HAPPEN UP FRONT THERE AND ARE TESTED IN THIS ORDER; no op is two of
    /// them, so the ordering is not observable.
    #[must_use]
    pub fn of_fusible_op(fusible_op: &Op) -> Option<Forwardings> {
        if let Some(mac_op) = MacOp::of(fusible_op) {
            return Some(Forwardings {
                op_a: mac_op.op_a.forwarding.clone(),
                op_b: mac_op.op_b.forwarding.clone(),
                op_c: mac_op.op_c.forwarding.clone(),
                result: mac_op.result.forwarding.clone(),
                result_precision: mac_op.result.precision,
            });
        }
        if let Some(binary_op) = BinaryOp::of(fusible_op) {
            return Some(Forwardings {
                op_a: binary_op.op_a.forwarding.clone(),
                op_b: binary_op.op_b.forwarding.clone(),
                op_c: Vec::new(),
                result: binary_op.result.forwarding.clone(),
                result_precision: binary_op.result.precision,
            });
        }
        let unary_op = UnaryOp::of(fusible_op)?;
        Some(Forwardings {
            op_a: unary_op.op_a.forwarding.clone(),
            op_b: Vec::new(),
            op_c: Vec::new(),
            result: unary_op.result.forwarding.clone(),
            result_precision: unary_op.result.precision,
        })
    }
}

/// WHETHER THE FUSIBLE OP TOOK THE TRIVIAL OP'S FORWARDINGS — `LogicalResult`, which here reports
/// whether the fusion fits ONE instruction rather than an error.
///
/// ⛔ DECLINING IS NOT FAILING, and the crate forbids spelling it as an error anyway: e480 abandons
/// this one fusion and its caller walks on (`:531-538`, `:570-574`). Same shape as
/// [`super::lightweight_simplification::Simplified`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum Updated {
    /// `LogicalResult::success()` — `ResultForwarding` now holds the trivial op's forwardings.
    Rewritten,
    /// `LogicalResult::failure()` — more than one LRF in the fusion, or an op that is none of the
    /// three compute classes.
    NotFused,
}

/// Replaces: e228_updateFusibleOp
///
/// Writes the trivial op's whole forwarding list into the fusible op's `ResultForwarding`, refusing
/// when that list names more than one LRF (`:387-429`).
///
/// ⭐ THE EFFECT IS THE PORT: `setResultForwardingAttr` on whichever of the three op classes it is.
/// ⛔ *"we cannot generate a single instruction using the current ISA"* (`:407-408`) is the refusal —
/// the count is of the TRIVIAL list only; the fusible list is never counted.
/// ⛔ `if (val == none) continue` (`:398`) IS DEAD: `stringify(none)` is `"none"`, which does not hold
/// `"lrf"`, so the skip changes no count.
/// ⛔ `mlir::cast<SentientComputePortAttr>(attr)` (`:397`) IS A HARD CAST that aborts on any other
/// attribute; a `Vec<Port>` makes that unrepresentable.
pub fn update_fusible_op(fusible_op: &mut Op, value_forwardings_for_trivial: &[Port]) -> Updated {
    let num_lrf_in_trivial = value_forwardings_for_trivial
        .iter()
        .filter(|&&port| matches!(port, Port::Lrf(_)))
        .count();
    if num_lrf_in_trivial > 1 {
        return Updated::NotFused;
    }
    match fusible_op {
        Op::Sentient(
            sentient::Op::VectorMac { result, .. }
            | sentient::Op::VectorBinary { result, .. }
            | sentient::Op::VectorUnary { result, .. },
        ) => {
            result.forwarding = value_forwardings_for_trivial.to_vec();
            Updated::Rewritten
        }
        _ => Updated::NotFused,
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Dd2;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::dataflow_ir::{GroupId, OpIndex, ProgramName, Units};
    use crate::islands::sentient::dialects::Val;
    use crate::islands::sentient::dialects::sentient::{
        Binary, FmaMode, LrfIndex, RegType, UnrollFactor,
    };
    use crate::islands::sentient::{ProgramUnit, ProgramUnits};
    use crate::units::{DfirUnit, Residency};

    /// A model, so the program is typed; nothing here reads it.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct AnyModel;
    impl Model for AnyModel {
        const QUERY_HEADS: u32 = 32;
        const KV_HEADS: u32 = 8;
        const HEAD_DIM: u32 = 64;
        const HIDDEN: u32 = 2048;
        const LAYERS: u32 = 40;
        const FFN: u32 = 8192;
        const VOCAB: u32 = 49152;
    }

    /// A decode rung, for the same reason.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct AnyRung;
    impl Workload for AnyRung {
        const ROWS: u32 = 1;
        const ACTIVE_CAP: u32 = 64;
    }

    /// One program whose only unit is on `kind` and holds `body`.
    fn program_on(kind: DfirUnit, body: Vec<Op>) -> Program<Dd2, AnyModel, AnyRung> {
        Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            preamble: Vec::new(),
            units: ProgramUnits::of(
                ProgramUnit {
                    on: Units::one(kind, Val(0)),
                    precision: None,
                    body,
                    arch: core::marker::PhantomData,
                },
                Vec::new(),
            ),
            bound: core::marker::PhantomData,
        }
    }

    /// An operand reading `port` and forwarded on to `forwarding`.
    fn operand(port: Port, forwarding: &[Port]) -> Operand {
        Operand {
            forwarding: forwarding.to_vec(),
            ..Operand::from(port)
        }
    }

    /// A result bundle forwarded to `forwarding`, at `precision`.
    fn result(forwarding: &[Port], precision: Precision) -> ResultPorts {
        ResultPorts {
            forwarding: forwarding.to_vec(),
            precision,
            unroll_incr: false,
        }
    }

    /// `sentient.vector_mac` — `op_a * op_b + op_c`.
    fn mac(op_a: Operand, op_b: Operand, op_c: Operand, result: ResultPorts) -> Op {
        Op::Sentient(sentient::Op::VectorMac {
            mask: None,
            xrf_write_ptr: None,
            xrf_read_ptr: None,
            results: vec![Val(10)],
            op_a,
            op_b,
            op_c,
            result,
            mode: FmaMode::FusedMulAdd,
            compute_precision: Precision::Fp16,
            fold_mode: None,
            unroll_factor: UnrollFactor::X1,
            xrf_read_incr: 0,
            xrf_write_incr: 0,
            data_transfer_only: false,
            is_data_weight: None,
            dbg_name: None,
        })
    }

    /// `sentient.vector_binary` — `op_a <op> op_b`.
    fn binary(op_a: Operand, op_b: Operand, op: sentient::BinaryOp, result: ResultPorts) -> Op {
        Op::Sentient(sentient::Op::VectorBinary {
            mask: Val(2),
            op_a,
            op_b,
            binary_op: Binary::Plain(op),
            result,
            compute_precision: Precision::Fp16,
            fold_mode: None,
            unroll_factor: UnrollFactor::X1,
            dbg_name: None,
        })
    }

    /// `sentient.vector_unary` — `<op> op_a`.
    fn unary(op_a: Operand, op: sentient::UnaryOp, result: ResultPorts) -> Op {
        Op::Sentient(sentient::Op::VectorUnary {
            mask: Val(2),
            op_a,
            unary_op: op,
            result,
            compute_precision: Precision::Fp16,
            fold_mode: None,
            unroll_factor: UnrollFactor::X1,
            dbg_name: None,
        })
    }

    /// `sentient.scalar_constant` — one of the two ops the walk steps over.
    fn constant() -> Op {
        Op::Sentient(sentient::Op::ScalarConstant {
            value: 0,
            result: Val(3),
            reg_locale: RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    /// `dataflow.get_unit` — the other.
    fn get_unit() -> Op {
        Op::Dataflow(dataflow::Op::GetUnit {
            result: Val(4),
            residency: Residency::Global,
            unit: DfirUnit::L3lu,
            num_folds: None,
            reg_locale: None,
        })
    }

    /// e223: a null `ArrayAttr` contributes nothing and a present one appends in order (`:94-98`).
    #[test]
    fn an_absent_forwarding_list_appends_nothing_and_a_present_one_appends_in_order() {
        let mut value_forwardings = vec![Port::West];
        append_to_vector(&mut value_forwardings, &[]);
        assert_eq!(value_forwardings, vec![Port::West]);
        append_to_vector(
            &mut value_forwardings,
            &[Port::Lrf(LrfIndex::L0), Port::Zero],
        );
        assert_eq!(
            value_forwardings,
            vec![Port::West, Port::Lrf(LrfIndex::L0), Port::Zero]
        );
    }

    /// e224: the walk steps over a `scalar_constant` and a `get_unit` to reach the mac — and a
    /// reduction `vector_unary` stops it unless it is queued for deletion, which is the fall-through
    /// at `:128-136`.
    #[test]
    fn the_walk_steps_over_bookkeeping_and_a_reduction_stops_it_unless_it_is_being_deleted() {
        let block = vec![
            mac(
                operand(Port::West, &[]),
                operand(Port::West, &[]),
                operand(Port::Zero, &[]),
                result(&[], Precision::Fp16),
            ),
            constant(),
            get_unit(),
            unary(
                operand(Port::West, &[]),
                sentient::UnaryOp::ReductionAdd,
                result(&[], Precision::Fp16),
            ),
            binary(
                operand(Port::Lrf(LrfIndex::L0), &[Port::West]),
                operand(Port::Zero, &[]),
                sentient::BinaryOp::Or,
                result(&[], Precision::Fp16),
            ),
        ];
        assert_eq!(
            find_fusible_op(InBlock::at(4), &block, &[], FoldAbMode::Off),
            None
        );
        assert_eq!(
            find_fusible_op(InBlock::at(4), &block, &[InBlock::at(3)], FoldAbMode::Off),
            Some(InBlock::at(0))
        );
    }

    /// e225: `lrf0 * one + zero` forwarding `opA` is trivial (`:167-172`); an accumulating mac with no
    /// zero operand is not (`:159-166`).
    #[test]
    fn a_mac_forwarding_an_lrf_through_a_multiply_by_one_is_trivial() {
        let forwarding_mac = mac(
            operand(Port::Lrf(LrfIndex::L0), &[Port::West]),
            operand(Port::One, &[]),
            operand(Port::Zero, &[]),
            result(&[], Precision::Fp16),
        );
        assert!(is_op_trivial_mac_op(MacOp::of(&forwarding_mac).unwrap()));
        let accumulating_mac = mac(
            operand(Port::Lrf(LrfIndex::L0), &[Port::West]),
            operand(Port::One, &[]),
            operand(Port::Lrf(LrfIndex::L1), &[]),
            result(&[], Precision::Fp16),
        );
        assert!(!is_op_trivial_mac_op(MacOp::of(&accumulating_mac).unwrap()));
    }

    /// e226: `lrf0 or0 zero` forwarding `opA` is trivial (`:192-196`); `and0` never is (`:185`).
    #[test]
    fn only_an_or_with_zero_forwarding_its_lrf_operand_is_trivial() {
        let or_zero = binary(
            operand(Port::Lrf(LrfIndex::L0), &[Port::West]),
            operand(Port::Zero, &[]),
            sentient::BinaryOp::Or,
            result(&[], Precision::Fp16),
        );
        assert!(is_op_trivial_binary_op(BinaryOp::of(&or_zero).unwrap()));
        let and_zero = binary(
            operand(Port::Lrf(LrfIndex::L0), &[Port::West]),
            operand(Port::Zero, &[]),
            sentient::BinaryOp::And,
            result(&[], Precision::Fp16),
        );
        assert!(!is_op_trivial_binary_op(BinaryOp::of(&and_zero).unwrap()));
    }

    /// e227: a `vector_unary` fills `opA`, the result and the precision, leaving `opB` and `opC` at
    /// the null `ArrayAttr` the reference never assigns (`:369-372`); a non-compute op is the
    /// `emitError` arm (`:373-377`).
    #[test]
    fn a_unary_contributes_op_a_and_the_result_and_a_non_compute_op_contributes_nothing() {
        let fusible = unary(
            operand(Port::Lrf(LrfIndex::L0), &[Port::West]),
            sentient::UnaryOp::Rec,
            result(&[Port::Lrf(LrfIndex::L1)], Precision::Int8),
        );
        assert_eq!(
            Forwardings::of_fusible_op(&fusible),
            Some(Forwardings {
                op_a: vec![Port::West],
                op_b: Vec::new(),
                op_c: Vec::new(),
                result: vec![Port::Lrf(LrfIndex::L1)],
                result_precision: Precision::Int8,
            })
        );
        assert_eq!(Forwardings::of_fusible_op(&constant()), None);
    }

    /// e228: two LRFs in the trivial op's forwardings cannot be one instruction and the fusible op is
    /// left alone (`:409-412`); one LRF becomes its `ResultForwarding` (`:414-427`).
    #[test]
    fn a_second_lrf_refuses_the_fusion_and_one_becomes_the_result_forwarding() {
        let mut fusible = mac(
            operand(Port::West, &[]),
            operand(Port::West, &[]),
            operand(Port::Zero, &[]),
            result(&[], Precision::Fp16),
        );
        let two_lrfs = [Port::Lrf(LrfIndex::L0), Port::Lrf(LrfIndex::L1)];
        let before = fusible.clone();
        assert_eq!(
            update_fusible_op(&mut fusible, &two_lrfs),
            Updated::NotFused
        );
        assert_eq!(fusible, before);
        let one_lrf = [Port::Lrf(LrfIndex::L0), Port::West];
        assert_eq!(
            update_fusible_op(&mut fusible, &one_lrf),
            Updated::Rewritten
        );
        assert_eq!(
            fusible,
            mac(
                operand(Port::West, &[]),
                operand(Port::West, &[]),
                operand(Port::Zero, &[]),
                result(&one_lrf, Precision::Fp16),
            )
        );
    }

    /// A `vector_mac` carrying `fold_mode` — the attribute e480's first two gates read.
    fn mac_folded(fold_mode: Option<sentient::FoldMode>) -> Op {
        let mut op = mac(
            operand(Port::West, &[]),
            operand(Port::West, &[]),
            operand(Port::Zero, &[]),
            result(&[], Precision::Fp16),
        );
        if let Op::Sentient(sentient::Op::VectorMac { fold_mode: on, .. }) = &mut op {
            *on = fold_mode;
        }
        op
    }

    /// e385: the four lists flatten onto the running list IN OPERAND ORDER, and the absent `opC` of a
    /// `vector_binary` contributes nothing (`:100-105`).
    #[test]
    fn e385_flattens_the_four_forwarding_lists_in_operand_order() {
        let mut value_forwardings = vec![Port::One];
        append_value_forwarding(
            &mut value_forwardings,
            &Forwardings {
                op_a: vec![Port::West],
                op_b: vec![Port::Lrf(LrfIndex::L0)],
                op_c: Vec::new(),
                result: vec![Port::Zero],
                result_precision: Precision::Fp16,
            },
        );
        assert_eq!(
            value_forwardings,
            vec![Port::One, Port::West, Port::Lrf(LrfIndex::L0), Port::Zero]
        );
    }

    /// e387: a trivial `vector_binary` hands back its own three lists and its precision; a
    /// `vector_mac` that computes something and a `vector_unary` — which is never trivial (`:334`) —
    /// hand back nothing.
    #[test]
    fn e387_fills_only_for_a_trivial_mac_or_binary() {
        let trivial = binary(
            operand(Port::Lrf(LrfIndex::L0), &[Port::West]),
            operand(Port::Zero, &[]),
            sentient::BinaryOp::Or,
            result(&[Port::One], Precision::Int8),
        );
        assert_eq!(
            Forwardings::of_trivial_op(&trivial),
            Some(Forwardings {
                op_a: vec![Port::West],
                op_b: Vec::new(),
                op_c: Vec::new(),
                result: vec![Port::One],
                result_precision: Precision::Int8,
            })
        );
        let computing_mac = mac(
            operand(Port::West, &[]),
            operand(Port::West, &[]),
            operand(Port::Zero, &[]),
            result(&[], Precision::Fp16),
        );
        assert_eq!(Forwardings::of_trivial_op(&computing_mac), None);
        let never_trivial = unary(
            operand(Port::Lrf(LrfIndex::L0), &[Port::West]),
            sentient::UnaryOp::Rec,
            result(&[], Precision::Fp16),
        );
        assert_eq!(Forwardings::of_trivial_op(&never_trivial), None);
    }

    /// A `vector_binary` at `unroll_factor`, whose `opA` increments per unrolled copy.
    fn binary_unrolled(factor: sentient::UnrollFactor, incr_op_a: bool) -> Op {
        let mut op = binary(
            Operand {
                unroll_incr: incr_op_a,
                ..operand(Port::Lrf(LrfIndex::L0), &[Port::West])
            },
            operand(Port::Zero, &[]),
            sentient::BinaryOp::Or,
            result(&[], Precision::Fp16),
        );
        if let Op::Sentient(sentient::Op::VectorBinary { unroll_factor, .. }) = &mut op {
            *unroll_factor = factor;
        }
        op
    }

    /// e386 — the eligible shape passes, and each of the four refusals fires on its own: a fold mode
    /// the other op does not carry and is not `fold_A`, a fusible list naming a port the TRIVIAL op
    /// does not read, an LRF on both sides, and a mismatched unroll factor.
    #[test]
    fn e386_admits_the_agreeing_pair_and_refuses_each_disagreement() {
        let trivial = binary_unrolled(sentient::UnrollFactor::X1, false);
        let fusible = mac(
            operand(Port::West, &[]),
            operand(Port::West, &[]),
            operand(Port::Zero, &[]),
            result(&[], Precision::Fp16),
        );
        assert!(is_eligible_to_fuse(
            &[Port::West],
            &[Port::Zero],
            &trivial,
            &fusible
        ));
        // `fold_B` on the trivial op alone (`:234-243`).
        assert!(!is_eligible_to_fuse(
            &[Port::West],
            &[Port::Zero],
            &mac_folded(Some(sentient::FoldMode::FoldB)),
            &fusible
        ));
        // `Port::One` is neither `opA` (`lrf0`) nor `opB` (`zero`) of the trivial binary (`:257-261`).
        assert!(!is_eligible_to_fuse(
            &[Port::West],
            &[Port::One],
            &trivial,
            &fusible
        ));
        // An LRF in the trivial list forbids ANY LRF in the fusible list (`:272-281`).
        assert!(!is_eligible_to_fuse(
            &[Port::Lrf(LrfIndex::L0)],
            &[Port::Lrf(LrfIndex::L1)],
            &trivial,
            &fusible
        ));
        // ⛔ x2 AGAINST THE MAC'S x1 (`:309-310`) — and at x2 the increments must agree too.
        assert!(!is_eligible_to_fuse(
            &[Port::West],
            &[Port::Zero],
            &binary_unrolled(sentient::UnrollFactor::X2, false),
            &fusible
        ));
    }

    /// e480: a `fold_AB_B` head is refused before anything is asked of the op (`:449-450`), and the
    /// whole chain now runs — a trivial `lrf0 or0 zero` forwarding `west` hands that forwarding to the
    /// `vector_mac` before it and is queued for deletion. ⭐ THE GATE ORDER IS THE PORT.
    #[test]
    fn e480_a_fold_ab_b_head_is_refused_and_a_trivial_forward_lands_on_the_op_before_it() {
        let mut to_be_deleted = Vec::new();
        let mut block = vec![mac_folded(Some(sentient::FoldMode::FoldAbB))];

        assert_eq!(
            fuse_store_and_forward(InBlock::at(0), &mut block, &mut to_be_deleted),
            Fused::No
        );
        assert_eq!(to_be_deleted, Vec::new());

        let mut fused = vec![
            mac_folded(None),
            binary(
                operand(Port::Lrf(LrfIndex::L0), &[Port::West]),
                operand(Port::Zero, &[]),
                sentient::BinaryOp::Or,
                result(&[], Precision::Fp16),
            ),
        ];
        assert_eq!(
            fuse_store_and_forward(InBlock::at(1), &mut fused, &mut to_be_deleted),
            Fused::Yes
        );
        assert_eq!(to_be_deleted, vec![InBlock::at(1)]);
        assert_eq!(
            fused[0],
            mac(
                operand(Port::West, &[]),
                operand(Port::West, &[]),
                operand(Port::Zero, &[]),
                result(&[Port::West], Precision::Fp16),
            )
        );
    }

    /// e541 — a PE unit's trivial forward is fused away and the op it fused into keeps the
    /// forwarding, while a unit that is neither SFP nor PE is left untouched. ⭐ THE UNIT FILTER IS
    /// THE PORT.
    #[test]
    fn e541_walks_only_the_sfp_and_pe_units() {
        let body = || {
            vec![
                mac_folded(None),
                binary(
                    operand(Port::Lrf(LrfIndex::L0), &[Port::West]),
                    operand(Port::Zero, &[]),
                    sentient::BinaryOp::Or,
                    result(&[], Precision::Fp16),
                ),
            ]
        };
        let mut skipped = program_on(DfirUnit::L3lu, body());
        run_on_operation(&mut skipped);
        assert_eq!(
            skipped.units.iter().next().expect("the one unit").body,
            body()
        );

        let mut walked = program_on(DfirUnit::Pe, body());
        run_on_operation(&mut walked);
        assert_eq!(
            walked.units.iter().next().expect("the one unit").body,
            vec![mac(
                operand(Port::West, &[]),
                operand(Port::West, &[]),
                operand(Port::Zero, &[]),
                result(&[Port::West], Precision::Fp16),
            )]
        );
    }
}

/// Replaces: e385_appendValueForwarding
///
/// Flattens one op's four forwarding lists onto a running list, in operand order (`:100-105`).
///
/// ⛔ `result_forwarding` COMES LAST, AFTER `opC` — and e480 appends the FUSIBLE op's result
/// forwarding a second time onto the trivial list (`:497`), so order is observable.
pub fn append_value_forwarding(value_forwardings: &mut Vec<Port>, forwardings: &Forwardings) {
    append_to_vector(value_forwardings, &forwardings.op_a);
    append_to_vector(value_forwardings, &forwardings.op_b);
    append_to_vector(value_forwardings, &forwardings.op_c);
    append_to_vector(value_forwardings, &forwardings.result);
}

/// THE UNROLL AGREEMENT A TRIVIAL OP CONTRIBUTES — `getUnrollFactorVal()` with the four
/// `unrollIncrOp*`/`unrollIncrResult` flags e386 reads off it (`:229-231`).
#[derive(Debug, Clone, Copy)]
struct Unroll {
    /// `op_unroll_factor`, which starts at `1`.
    factor: sentient::UnrollFactor,
    /// `op_unrollIncrOpA`.
    incr_op_a: bool,
    /// `op_unrollIncrOpB`.
    incr_op_b: bool,
    /// `op_unrollIncrOpC` — never set for a `vector_binary`, which has no `opC`.
    incr_op_c: bool,
    /// `op_unrollIncrResult`.
    incr_result: bool,
}

impl Unroll {
    /// `unsigned int op_unroll_factor = 1;` and the four `false`s beside it (`:229-231`) — what a
    /// trivial op that is neither a `vector_mac` nor a `vector_binary` leaves them at.
    const NONE: Unroll = Unroll {
        factor: sentient::UnrollFactor::X1,
        incr_op_a: false,
        incr_op_b: false,
        incr_op_c: false,
        incr_result: false,
    };

    /// `op_unrollIncrOpA || op_unrollIncrOpB || op_unrollIncrOpC || op_unrollIncrResult` (`:310-311`).
    const fn any_incr(self) -> bool {
        self.incr_op_a || self.incr_op_b || self.incr_op_c || self.incr_result
    }
}

/// Replaces: e386_isEligibleToFuse
///
/// Whether the trivial op's forwardings may be folded into the fusible op's: the two must agree on
/// fold mode, the fusible list may only name ports the trivial op actually reads, the two lists may
/// not collide and may not both name an LRF, and the unroll factors must match (`:219-315`).
///
/// ⛔ THE FUSIBLE LIST IS CHECKED AGAINST THE **TRIVIAL** OP'S PORTS (`:245-251`, `:257-261`), not the
/// fusible op's, and a `vector_unary` trivial op has no arm at all — so it contributes the `1`/`false`
/// defaults and skips that check entirely.
/// ⛔ `stringifySentientComputePort(port).contains("lrf")` IS `Port::Lrf(_)`, as in e225: one LRF port
/// in the trivial list forbids ANY LRF in the fusible list, and a non-LRF forbids only its own repeat.
/// ⭐ THE FACTORS ARE COMPARED AS THE ATTRIBUTE, NOT THE PARSED `unsigned`: `x1..x8` map one-to-one
/// onto their counts, so `!=` on the enum is `getUnrollFactorVal() !=` without the parse.
#[must_use]
pub fn is_eligible_to_fuse(
    value_forwardings_for_trivial: &[Port],
    value_forwardings_for_fusible: &[Port],
    trivial_op: &Op,
    fusible_op: &Op,
) -> bool {
    // "Default fold is fold_A": one op carrying a mode the other does not may only carry `fold_A`,
    // and two carrying one must carry the same (`:234-243`).
    let trivial_op_fold = fold_mode_attribute_if_exists(trivial_op);
    let fusible_op_fold = fold_mode_attribute_if_exists(fusible_op);
    match (trivial_op_fold, fusible_op_fold) {
        (Some(trivial), None) if trivial != sentient::FoldMode::FoldA => return false,
        (None, Some(fusible)) if fusible != sentient::FoldMode::FoldA => return false,
        (Some(trivial), Some(fusible)) if trivial != fusible => return false,
        _ => {}
    }

    let trivial_unroll = if let Some(mac_op) = MacOp::of(trivial_op) {
        if value_forwardings_for_fusible.iter().any(|port| {
            *port != mac_op.op_a.port && *port != mac_op.op_b.port && *port != mac_op.op_c.port
        }) {
            return false;
        }
        Unroll {
            factor: mac_op.unroll_factor,
            incr_op_a: mac_op.op_a.unroll_incr,
            incr_op_b: mac_op.op_b.unroll_incr,
            incr_op_c: mac_op.op_c.unroll_incr,
            incr_result: mac_op.result.unroll_incr,
        }
    } else if let Some(binary_op) = BinaryOp::of(trivial_op) {
        if value_forwardings_for_fusible
            .iter()
            .any(|port| *port != binary_op.op_a.port && *port != binary_op.op_b.port)
        {
            return false;
        }
        Unroll {
            factor: binary_op.unroll_factor,
            incr_op_a: binary_op.op_a.unroll_incr,
            incr_op_b: binary_op.op_b.unroll_incr,
            incr_op_c: false,
            incr_result: binary_op.result.unroll_incr,
        }
    } else {
        Unroll::NONE
    };

    for port in value_forwardings_for_trivial {
        if matches!(port, Port::Lrf(_)) {
            if value_forwardings_for_fusible
                .iter()
                .any(|fusible| matches!(fusible, Port::Lrf(_)))
            {
                return false;
            }
        } else if value_forwardings_for_fusible.contains(port) {
            return false;
        }
    }

    // ⛔ THREE ARMS HERE, THE `vector_unary` INCLUDED (`:299-308`), and only the factor and the
    // result increment are read off the fusible op.
    let (fusible_factor, fusible_incr_result) = if let Some(mac_op) = MacOp::of(fusible_op) {
        (mac_op.unroll_factor, mac_op.result.unroll_incr)
    } else if let Some(binary_op) = BinaryOp::of(fusible_op) {
        (binary_op.unroll_factor, binary_op.result.unroll_incr)
    } else if let Some(unary_op) = UnaryOp::of(fusible_op) {
        (unary_op.unroll_factor, unary_op.result.unroll_incr)
    } else {
        (sentient::UnrollFactor::X1, false)
    };
    if trivial_unroll.factor != fusible_factor {
        return false;
    }
    if trivial_unroll.factor != sentient::UnrollFactor::X1
        && fusible_incr_result != trivial_unroll.any_incr()
    {
        return false;
    }

    true
}

impl Forwardings {
    /// Replaces: e387_fillOperandForTrivialOp
    ///
    /// The forwardings and result precision of a `vector_mac` or `vector_binary`, and [`None`] unless
    /// that op computes nothing but a forward (`:318-336`).
    ///
    /// ⛔ [`None`] IS THE REFERENCE'S `false`, WHICH IT ALSO RETURNS WITH THE OUT-PARAMS FILLED when
    /// the op is a compute that simply is not trivial — e480 abandons the fusion either way, so the
    /// two are one answer here.
    /// ⛔ *"there is no trivial UnaryOp, TernaryOp in Sentient"* (`:334`) — hence two arms where
    /// [`Forwardings::of_fusible_op`] has three.
    #[must_use]
    pub fn of_trivial_op(op: &Op) -> Option<Forwardings> {
        if let Some(mac_op) = MacOp::of(op) {
            return is_op_trivial_mac_op(mac_op).then(|| Forwardings {
                op_a: mac_op.op_a.forwarding.clone(),
                op_b: mac_op.op_b.forwarding.clone(),
                op_c: mac_op.op_c.forwarding.clone(),
                result: mac_op.result.forwarding.clone(),
                result_precision: mac_op.result.precision,
            });
        }
        let binary_op = BinaryOp::of(op)?;
        is_op_trivial_binary_op(binary_op).then(|| Forwardings {
            op_a: binary_op.op_a.forwarding.clone(),
            op_b: binary_op.op_b.forwarding.clone(),
            op_c: Vec::new(),
            result: binary_op.result.forwarding.clone(),
            result_precision: binary_op.result.precision,
        })
    }
}

/// `getDbgNameAttr(op)` for the three compute classes — the two names `"SAFF("` is built from.
fn dbg_name_of(op: &Op) -> Option<&str> {
    let Op::Sentient(
        sentient::Op::VectorMac { dbg_name, .. }
        | sentient::Op::VectorBinary { dbg_name, .. }
        | sentient::Op::VectorUnary { dbg_name, .. },
    ) = op
    else {
        return None;
    };
    dbg_name.as_deref()
}

/// `setDbgNameAttr(op, name)`.
///
/// ⛔ ONLY EVER WITH A NAME: `setDbgNameAttr(op, nullptr)` REMOVES the attribute
/// (`DataflowOpInterfaces.cpp:49`), which is why e480 guards on the `StringAttr` (`:539-542`).
fn set_dbg_name(op: &mut Op, name: String) {
    if let Op::Sentient(
        sentient::Op::VectorMac { dbg_name, .. }
        | sentient::Op::VectorBinary { dbg_name, .. }
        | sentient::Op::VectorUnary { dbg_name, .. },
    ) = op
    {
        *dbg_name = Some(name);
    }
}

/// WHETHER ONE CANDIDATE FUSED — `LogicalResult`, which here reports whether this store and the
/// forward before it became one instruction rather than an error.
///
/// ⛔ DECLINING IS NOT FAILING, the same shape as [`Updated`]: e541 walks on either way (`:570-574`).
/// ⛔ AND [`Self::No`] IS NOT ALWAYS "NOTHING HAPPENED" — in `fold_AB` mode the FIRST fusible op keeps
/// the forwardings [`update_fusible_op`] wrote into it when the second round then declines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum Fused {
    /// `LogicalResult::success()` — the fusible op took the forwardings and `op` is queued for
    /// deletion.
    Yes,
    /// `LogicalResult::failure()` — one of the gates below declined.
    No,
}

/// Replaces: e480_FuseStoreAndForward
///
/// Fuses the trivial compute at `op` into the compute before it: that op takes over `op`'s
/// forwardings and gains a `SAFF(…)` name, and `op` — with its `fold_AB_B` partner — is queued for
/// deletion.
///
/// ⛔ THE SECOND `fillOperandForTrivialOp` IS HANDED THE **SAME** `op` (`:456-461`), not
/// `op->getNextNode()`, so the `fold_AB_B` half's own forwardings are never read — ported as written.
/// ⛔ BOTH RESULT PRECISIONS ARE ONE VARIABLE EACH that the fold round overwrites (`:459`, `:487`), so
/// the equality at `:493` is the SECOND fusible op's against the trivial op's.
/// ⛔ AND A DECLINING FOLD ROUND LEAVES THE FIRST FUSIBLE OP REWRITTEN — see [`Fused`].
pub fn fuse_store_and_forward(
    op: InBlock,
    block: &mut Vec<Op>,
    to_be_deleted: &mut Vec<InBlock>,
) -> Fused {
    let next = InBlock::at(op.index() + 1);
    // COPIED OUT, not held: the reference keeps `op` as a pointer across `setResultForwardingAttr`
    // on the fusible op, which is a second path into the same block. e541 hands positions from the
    // block it walks, so an absent one cannot arise.
    let Some(trivial_op) = block.get(op.index()).cloned() else {
        return Fused::No;
    };
    let op_a_fold = fold_mode_attribute_if_exists(&trivial_op);
    // `getFoldModeAttributeIfExists(op->getNextNode())` — e248's own null check answers the last op.
    let op_b_fold = block
        .get(next.index())
        .and_then(fold_mode_attribute_if_exists);
    let fold_ab_mode = if op_a_fold == Some(sentient::FoldMode::FoldAbA)
        && op_b_fold == Some(sentient::FoldMode::FoldAbB)
    {
        FoldAbMode::On
    } else {
        FoldAbMode::Off
    };
    if op_a_fold == Some(sentient::FoldMode::FoldAbB) {
        return Fused::No;
    }

    let Some(trivial_first) = Forwardings::of_trivial_op(&trivial_op) else {
        return Fused::No;
    };
    let trivial_second = match fold_ab_mode {
        // `trivialOps.second` READS THE SAME OP, so it is that same answer in the second slot.
        FoldAbMode::On => match Forwardings::of_trivial_op(&trivial_op) {
            None => return Fused::No,
            second => second,
        },
        FoldAbMode::Off => None,
    };

    let Some(fusible_first) = find_fusible_op(op, block, to_be_deleted, fold_ab_mode) else {
        return Fused::No;
    };
    let fusible_second = match fold_ab_mode {
        // ⛔ DELIBERATE DIVERGENCE, NOT A LOST FUSION: `fusible_ops.second == op` (`:470`) tests the
        // wrong op, so a reference run that finds none carries on with the `fold_AB_B` half ITSELF
        // as its own fusible op — it writes the fold round's forwardings into that op (`:536`) and
        // then queues that very op for deletion (`:546`), so those forwardings leave the program.
        // Declining keeps both ops, and with them everything they forward.
        FoldAbMode::On => match find_fusible_op(next, block, to_be_deleted, FoldAbMode::On) {
            None => return Fused::No,
            second => second,
        },
        FoldAbMode::Off => None,
    };

    let Some(fusible_first_forwardings) = block
        .get(fusible_first.index())
        .and_then(Forwardings::of_fusible_op)
    else {
        return Fused::No;
    };
    let mut fusible_second_forwardings = None;
    if let Some(second) = fusible_second {
        let Some(filled) = block
            .get(second.index())
            .and_then(Forwardings::of_fusible_op)
        else {
            return Fused::No;
        };
        fusible_second_forwardings = Some(filled);
    }
    let fusible_precision = match &fusible_second_forwardings {
        Some(second) => second.result_precision,
        None => fusible_first_forwardings.result_precision,
    };
    let trivial_precision = match &trivial_second {
        Some(second) => second.result_precision,
        None => trivial_first.result_precision,
    };
    if fusible_precision != trivial_precision {
        return Fused::No;
    }

    let mut fusible_values_first = Vec::new();
    append_value_forwarding(&mut fusible_values_first, &fusible_first_forwardings);
    let mut trivial_values_first = Vec::new();
    append_value_forwarding(&mut trivial_values_first, &trivial_first);
    let mut fusible_values_second = Vec::new();
    let mut trivial_values_second = Vec::new();
    if let (Some(fusible), Some(trivial)) = (&fusible_second_forwardings, &trivial_second) {
        append_value_forwarding(&mut fusible_values_second, fusible);
        append_value_forwarding(&mut trivial_values_second, trivial);
    }

    let Some(fusible_op) = block.get(fusible_first.index()) else {
        return Fused::No;
    };
    if !is_eligible_to_fuse(
        &trivial_values_first,
        &fusible_values_first,
        &trivial_op,
        fusible_op,
    ) {
        return Fused::No;
    }
    if let Some(second) = fusible_second {
        // ⭐ HERE THE TRIVIAL OP IS `op->getNextNode()` (`:520`), unlike `:456`'s second fill.
        let (Some(next_op), Some(second_op)) = (block.get(next.index()), block.get(second.index()))
        else {
            return Fused::No;
        };
        if !is_eligible_to_fuse(
            &trivial_values_second,
            &fusible_values_second,
            next_op,
            second_op,
        ) {
            return Fused::No;
        }
    }

    append_to_vector(&mut trivial_values_first, &fusible_first_forwardings.result);
    if let Some(second) = &fusible_second_forwardings {
        append_to_vector(&mut trivial_values_second, &second.result);
    }

    let Some(fusible_op) = block.get_mut(fusible_first.index()) else {
        return Fused::No;
    };
    if update_fusible_op(fusible_op, &trivial_values_first) == Updated::NotFused {
        return Fused::No;
    }
    if let Some(second) = fusible_second {
        let Some(second_op) = block.get_mut(second.index()) else {
            return Fused::No;
        };
        if update_fusible_op(second_op, &trivial_values_second) == Updated::NotFused {
            return Fused::No;
        }
    }

    // `getNewDbgNameFromList("SAFF(", {fusible_ops.first, op})`, which is [`None`] unless BOTH are
    // named, and then leaves the fusible op's own name alone.
    let new_dbg_name = new_dbg_name_from_list(
        DbgNamePrefix::Saff,
        block.get(fusible_first.index()).and_then(dbg_name_of),
        &[dbg_name_of(&trivial_op)],
    );
    if let Some(name) = new_dbg_name
        && let Some(fusible_op) = block.get_mut(fusible_first.index())
    {
        set_dbg_name(fusible_op, name);
    }

    to_be_deleted.push(op);
    if fold_ab_mode == FoldAbMode::On {
        to_be_deleted.push(next);
    }
    // `std::sort` + `std::unique` + `erase` — the queue is an ordered set of positions.
    to_be_deleted.sort_unstable();
    to_be_deleted.dedup();
    Fused::Yes
}

/// `-dcc-store-and-forward-fusion-disable`, `cl::init(false)` (`:50-53`) — a `dcc-opt` command-line
/// flag, not a program property, and this crate has no flags.
const DISABLE_THIS_PASS: bool = false;

/// `opts_.OptLevel == 0` (`:568`) — the PIPELINE's optimisation level, which is `2` by default
/// (`dcc/tools/Options/dcc-pass-option.h:63-65`, copied into the common options at
/// `dcc/tools/dcc-standalone/dcc-standalone-main.cpp:701`), so the shipped pipeline never reaches the
/// `haveIbuffSpace` half of the `&&`.
const OPT_LEVEL_ZERO: bool = false;

/// ONE BLOCK AND EVERY BLOCK NESTED IN IT — `unit_op.walk([&](mlir::Operation *op) { … })`, whose
/// default order is post-order, with the deletions of each block applied when its own walk ends.
///
/// ⛔ THE QUEUE IS PER BLOCK, NOT PER MODULE, AND THAT IS A REPRESENTATION CHANGE. The reference keeps
/// one `std::vector<Operation *>` for the whole module and erases at the very end; an [`InBlock`] is
/// an index into ONE `Vec<Op>`, so a shared queue would collide and e480's own `sort`+`unique` would
/// merge unrelated positions. It is observationally identical because e480 reads only the op's own
/// block — and the erase runs in DESCENDING position order so the queued indices stay valid.
fn fuse_block(block: &mut Vec<Op>) {
    for at in 0..block.len() {
        for region in dialects::regions_mut(&mut block[at]) {
            fuse_block(region);
        }
    }
    let mut to_be_deleted = Vec::new();
    for at in 0..block.len() {
        // ⚠️ THE `LLVM_DEBUG` ON A SUCCESSFUL FUSION IS DROPPED (`:571-573`): it changes no IR.
        let _ = fuse_store_and_forward(InBlock::at(at), block, &mut to_be_deleted);
    }
    for op in to_be_deleted.iter().rev() {
        block.remove(op.index());
    }
}

/// Replaces: e541_runOnOperation
///
/// The pass entry: unless the flag disables it, fuse the trivial computes of every SFP and PE unit of
/// the module into the compute before them, and erase the ops that were fused away.
///
/// ⛔ THE MISSING ANALYSIS IS NAMED, NOT SUBSTITUTED FOR: `haveIbuffSpace` stays a `todo!` behind
/// [`OPT_LEVEL_ZERO`], whose value the pipeline fixes rather than this pass — so flipping that const
/// reaches the out-of-scope estimator instead of quietly skipping a unit.
/// ⭐ `getChildAnalysis<InstructionEstimator>(unit_op)` IS PER UNIT, and it is constructed even for a
/// unit the `&&` never asks anything of, exactly as the reference does (`:565-570`).
pub fn run_on_operation<A: Arch, M: Model, W: Workload>(program: &mut Program<A, M, W>) {
    if DISABLE_THIS_PASS {
        return;
    }
    for unit in program.units.iter_mut() {
        // `is_any_of(getUnitType(unit_op.getUnits()[0]), SFP, PE)` (`:561-563`).
        if !matches!(unit.on.kind().generic(), GenericComp::Sfp | GenericComp::Pe) {
            continue;
        }
        let mut instruction_estimator = OutOfScopeInstructionEstimator;
        if OPT_LEVEL_ZERO && instruction_estimator.have_ibuff_space(&unit.body) {
            continue;
        }
        fuse_block(&mut unit.body);
    }
}
