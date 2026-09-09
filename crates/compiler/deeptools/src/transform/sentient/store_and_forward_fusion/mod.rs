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

// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET — `e480_FuseStoreAndForward` (level 2) is what calls
// every item below and `e541_runOnOperation` (level 3) what walks the module, so until they land this
// file is reachable only from its own tests. CI runs clippy with `-D warnings`.
// ⭐ REMOVE THIS WITH e541.
#![allow(dead_code)]

use super::rematerialization_pass::InBlock;
use crate::islands::sentient::dialects::sentient::{Operand, Port, Precision, ResultPorts};
use crate::islands::sentient::dialects::{Op, dataflow, sentient};

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
    /// `getResultForwarding()` and `getResultPrecision()`.
    pub result: &'a ResultPorts,
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
                ..
            }) => Some(MacOp {
                op_a,
                op_b,
                op_c,
                result,
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
    /// `getResultForwarding()` and `getResultPrecision()`.
    pub result: &'a ResultPorts,
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
                ..
            }) => Some(BinaryOp {
                op_a,
                op_b,
                op: *binary_op,
                result,
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
    /// `getResultForwarding()` and `getResultPrecision()`.
    pub result: &'a ResultPorts,
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
                ..
            }) => Some(UnaryOp {
                op_a,
                op: *unary_op,
                result,
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
/// ⛔ [`None`] IS THE REFERENCE'S "HANDS BACK ITS OWN ARGUMENT", and that is a strict improvement:
/// e480's second guard tests `fusible_ops.second == op` (`:470`) after calling this on
/// `op->getNextNode()`, so its sentinel comparison can never fire. Absence cannot be misread.
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
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::Val;
    use crate::islands::sentient::dialects::sentient::{
        Binary, FmaMode, LrfIndex, RegType, UnrollFactor,
    };
    use crate::units::{DfirUnit, Residency};

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
}

// crustify:todo: e385_appendValueForwarding
//   authority : dcc/src/Transform/Sentient/StoreAndForwardFusion.cpp:100  (6 body lines, level 1)
//   original  : static void appendValueForwarding( std::vector<mlir::Attribute> &value_forwardings, mlir::ArrayAttr &opA_forwarding, mlir::ArrayAttr &opB_forwarding, mlir::ArrayAttr &opC_forwarding, mlir::ArrayAttr &result_forwarding)
//   calls     : e223_appendToVector

// crustify:todo: e386_isEligibleToFuse
//   authority : dcc/src/Transform/Sentient/StoreAndForwardFusion.cpp:219  (87 body lines, level 1)
//   original  : bool StoreAndForwardFusionPass::isEligibleToFuse( std::vector<mlir::Attribute> &value_forwardings_for_trivial, std::vector<mlir::Attribute> &value_forwardings_for_fusible, mlir::Operation *trivial_op, mlir::Operation *fusible_op)
//   calls     : e248_getFoldModeAttributeIfExists

// crustify:todo: e387_fillOperandForTrivialOp
//   authority : dcc/src/Transform/Sentient/StoreAndForwardFusion.cpp:318  (19 body lines, level 1)
//   original  : bool StoreAndForwardFusionPass::fillOperandForTrivialOp( mlir::Operation *op, mlir::ArrayAttr &opA_forwarding, mlir::ArrayAttr &opB_forwarding, mlir::ArrayAttr &opC_forwarding, mlir::ArrayAttr &result_forwarding, SentientPrecision &result_precision)
//   calls     : e225_isOpTrivialMacOp, e226_isOpTrivialBinaryOp

// crustify:todo: e480_FuseStoreAndForward
//   authority : dcc/src/Transform/Sentient/StoreAndForwardFusion.cpp:434  (119 body lines, level 2)
//   original  : LogicalResult StoreAndForwardFusionPass::FuseStoreAndForward( mlir::Operation *op, std::vector<mlir::Operation *> &to_be_deleted)
//   calls     : e223_appendToVector, e224_findFusibleOp, e227_fillOperandForFusibleOp, e228_updateFusibleOp, e248_getFoldModeAttributeIfExists, e385_appendValueForwarding, e386_isEligibleToFuse, e387_fillOperandForTrivialOp

// crustify:todo: e541_runOnOperation
//   authority : dcc/src/Transform/Sentient/StoreAndForwardFusion.cpp:555  (24 body lines, level 3)
//   original  : void StoreAndForwardFusionPass::runOnOperation()
//   calls     : e480_FuseStoreAndForward
