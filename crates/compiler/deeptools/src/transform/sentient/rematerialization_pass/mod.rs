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

//! `RematerializationPass.cpp` — 5 of the campaign's 656 units (dependency level(s) [0, 1, 2, 3]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e145_isCandidateForRematerialization` | 145 | 0 | 15 | `dcc/src/Transform/Sentient/RematerializationPass.cpp:60` |
//! | `e146_getLastUseWithinBlock` | 146 | 0 | 20 | `dcc/src/Transform/Sentient/RematerializationPass.cpp:82` |
//! | `e353_increasesOperandLiverange` | 353 | 1 | 52 | `dcc/src/Transform/Sentient/RematerializationPass.cpp:109` |
//! | `e464_runOn` | 464 | 2 | 42 | `dcc/src/Transform/Sentient/RematerializationPass.cpp:175` |
//! | `e525_runOnOperation` | 525 | 3 | 11 | `dcc/src/Transform/Sentient/RematerializationPass.cpp:163` |

// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET. `e525_runOnOperation` — the pass entry — has landed
// as [`run_on_operation`], and nothing in this crate calls it, so every item below is still reachable
// only from this file's own tests. CI runs clippy with `-D warnings`.
// ⭐ REMOVE THIS WHEN A PIPELINE CALLS `run_on_operation`: from then on an unused item here is a real
// defect.
#![allow(dead_code)]

use crate::arch::Arch;
use crate::islands::dataflow_ir::{ValueMapping, Values};
use crate::islands::sentient::Program;
use crate::islands::sentient::dialects::{
    self as dialects, Definitions, Op, Val, sentient, use_count,
};
use crate::model::Model;
use crate::transform::sentient::utils::{self, ConstKind, OpAt};
use crate::workload::Workload;

/// A VALUE SOME ENCLOSING REGION BINDS AS A RESULT — `DT_CHECK_MSG(!isa<BlockArgument>(v), "Function
/// should not be called on an iter arg")` (`:83-84`) AS THE PARAMETER TYPE.
///
/// ⛔ AN ITER ARG IS EXACTLY WHAT THE CHECK REFUSES, and a region argument is bound by no op — so
/// [`Definitions::of`] answering [`None`] IS the failed `dyn_cast<BlockArgument>`, and the witness
/// costs no second walk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Defined(Val);

impl Defined {
    /// `None` for a region argument, which is the one input the reference aborts on.
    #[must_use]
    pub fn of(val: Val, defs: Definitions<'_>) -> Option<Defined> {
        defs.of(val).map(|_| Defined(val))
    }

    /// The value itself.
    #[must_use]
    pub const fn val(self) -> Val {
        self.0
    }
}

/// WHERE AN OP SITS IN ONE BLOCK — what `Block::findAncestorOpInBlock` hands back, reduced to the one
/// fact `DominanceInfo` carries between two ops of the SAME block.
///
/// # ⭐⭐ THE `DominanceInfo` IS DROPPABLE MECHANISM, AND THAT IS PROVABLE
///
/// `dominates(a, b)` for `a` and `b` in one block is `a == b || a comes first`, so
/// `if (dom_info_->dominates(last_use, cand)) last_use = cand;` (`:96-97`) keeps whichever of the two
/// is LATER in the block — both having been mapped into `bb` by `findAncestorOpInBlock` before the
/// comparison. A position is the whole of what this pass asks the dominance tree, which is why
/// `new DominanceInfo(unit_op)` (`:168`) has nothing to answer here.
///
/// ⛔ AND IT IS ONLY COMPARABLE WITHIN THE BLOCK IT CAME FROM. Two positions from different blocks
/// are unrelated; the reference has the same restriction and discharges it the same way, by mapping
/// every operand into one block first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct InBlock(usize);

impl InBlock {
    /// Position `index` of a block.
    #[must_use]
    pub const fn at(index: usize) -> InBlock {
        InBlock(index)
    }

    /// The position, as an index into the block it came from.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

/// Replaces: e145_isCandidateForRematerialization
///
/// Whether `op` may be cloned before each of its uses: a `sentient.scalar_add` or `sentient.scalar_sub`
/// with a `sentient.scalar_constant` operand, or a `sentient.scalar_copy` of one (`:58-73`).
///
/// ⛔ TRAP: `sentient.scalar_mul` IS NOT A CANDIDATE. `Sentient_MulOp` is its own op class
/// (`SentientOps.td:816`) and the `dyn_cast` chain names only `AddOp`, `SubOp` and `CopyOp`.
///
/// ⭐ THE `!isa<BlockArgument>` GUARDS FUSE INTO THE LOOKUP — see [`is_constant`], which is also why
/// the `scalar_copy` arm is SAFE here and is a null deref in the reference.
#[must_use]
pub fn is_candidate_for_rematerialization(op: &Op, defs: Definitions<'_>) -> bool {
    match op {
        Op::Sentient(
            sentient::Op::ScalarAdd { lhs, rhs, .. } | sentient::Op::ScalarSub { lhs, rhs, .. },
        ) => is_constant(*lhs, defs) || is_constant(*rhs, defs),
        Op::Sentient(sentient::Op::ScalarCopy { input, .. }) => is_constant(*input, defs),
        // `return false;` (`:73`) — anything the three `dyn_cast`s decline.
        _ => false,
    }
}

/// `isa<sentient::ConstantOp>(v.getDefiningOp())`, with the `!isa<BlockArgument>(v)` guard folded in.
///
/// ⛔ THE REFERENCE'S `scalar_copy` ARM HAS NO GUARD OF ITS OWN (`:71-72`): it hands
/// `copy_op.getInp().getDefiningOp()` straight to `isa<>`, which dereferences a null `Operation *`
/// when the copy reads an iter arg. This port answers that arm's stated intent, *"a copy of a constant
/// op"* (`:58-59`), and a copy of an iter arg is `false` — a deliberate divergence from a crash.
fn is_constant(val: Val, defs: Definitions<'_>) -> bool {
    matches!(
        defs.of(val),
        Some(Op::Sentient(sentient::Op::ScalarConstant { .. }))
    )
}

/// Replaces: e146_getLastUseWithinBlock
///
/// The position in `block` of the LAST op whose subtree reads `v` — `findAncestorOpInBlock` for every
/// use of `v`, then the later of the two by dominance (`:82-101`).
///
/// ⭐ A USE NESTED IN A REGION IS A USE BY THE OP THAT ENCLOSES IT, which is exactly what
/// `bb->findAncestorOpInBlock(*use.getOwner())` returns; [`use_count`] descends into regions, so one
/// call per op of the block performs both the use walk and that mapping.
///
/// ⛔ `None` COVERS BOTH `nullptr` RETURNS: `v.use_empty()` (`:85`) and "every use lies outside this
/// block", which is `last_use` still null at `:100` after every `continue` (`:91`).
#[must_use]
pub fn last_use_within_block(v: Defined, block: &[Op]) -> Option<InBlock> {
    block
        .iter()
        .rposition(|op| use_count(v.val(), core::slice::from_ref(op)) > 0)
        .map(InBlock::at)
}

/// Replaces: e353_increasesOperandLiverange
///
/// Whether cloning `op` in front of `user` would keep one of its operands alive longer: it would if
/// `op` has no non-constant operand at all, or if `user` sits after such an operand's last use in the
/// block (`:109-161`).
///
/// ⛔⛔ `DominanceInfo` AND `findAncestorOpInBlock` ARE BOTH IN THE SIGNATURE, NOT THE BODY: `user` is
/// already the ancestor of the user within `op`'s own block ([`InBlock`]), and dominance between two
/// ops of ONE block is `user <= last_use`.
/// ⛔ TRAP: NO NON-CONSTANT OPERAND IS `true` (`:130`), not `false` — the reference's *"if any
/// non-constant operands have no uses, then rematerialization would extend their live range"*.
/// ⛔ TRAP: THE CONSTANT TEST HERE IS THE QUERY-MAP-AWARE `dcc::utils::isConstant<sentient::ConstantOp>`
/// (`Utils/Utils.cpp:424`), NOT the raw `isa` of [`is_candidate_for_rematerialization`].
/// ⛔ A REGION-ARGUMENT OPERAND IS SKIPPED rather than checked ([`Defined`] is the `!isa<BlockArgument>`
/// guard), and `DT_CHECK_MSG(operand_last_use, "Op itself is a use of its operands")` (`:154`) — which
/// cannot fire while `op` is in `block` — takes the conservative "extends" arm instead of aborting.
#[must_use]
pub fn increases_operand_liverange(
    op: &Op,
    user: InBlock,
    block: &[Op],
    defs: Definitions<'_>,
) -> bool {
    let operands: Vec<Val> = match op {
        Op::Sentient(sentient::Op::ScalarCopy { input, .. }) => vec![*input],
        Op::Sentient(
            sentient::Op::ScalarAdd { lhs, rhs, .. } | sentient::Op::ScalarSub { lhs, rhs, .. },
        ) => vec![*lhs, *rhs],
        // Every other op queues nothing, so the `empty()` arm below answers for it.
        _ => Vec::new(),
    };
    let to_check: Vec<Val> = operands
        .into_iter()
        .filter(|val| !utils::is_constant(*val, ConstKind::ScalarConstant, defs))
        .collect();
    if to_check.is_empty() {
        return true;
    }
    to_check.into_iter().any(|val| {
        Defined::of(val, defs).is_some_and(|defined| {
            last_use_within_block(defined, block).is_none_or(|last_use| user > last_use)
        })
    })
}

/// THE CANDIDATES OF ONE PROGRAM UNIT, PRE-ORDER — `unit_op.walk<WalkOrder::PreOrder>` (`:177-179`),
/// naming each by the one result it binds so that the answer survives the rewrites that follow.
fn candidates(block: &[Op], defs: Definitions<'_>, into: &mut Vec<Val>) {
    for op in block {
        if is_candidate_for_rematerialization(op, defs) {
            // `op->getResults()[0]` — all three candidate ops bind exactly one result.
            if let Some(&result) = dialects::results(op).first() {
                into.push(result);
            }
        }
        for region in dialects::regions_ref(op) {
            candidates(region, defs, into);
        }
    }
}

/// `result.getUses()` — every `(owner, use.getOperandNumber())` that reads `val`, in pre-order.
fn uses_of(
    block: &[Op],
    val: Val,
    enclosing: &mut Vec<(utils::InBlock, usize)>,
    into: &mut Vec<(OpAt, usize)>,
) {
    for (index, op) in block.iter().enumerate() {
        for (operand_number, operand) in dialects::operands(op).into_iter().enumerate() {
            if operand == val {
                into.push((OpAt::at(enclosing, utils::InBlock(index)), operand_number));
            }
        }
        for (region, inner) in dialects::regions_ref(op).into_iter().enumerate() {
            enclosing.push((utils::InBlock(index), region));
            uses_of(inner, val, enclosing, into);
            enclosing.pop();
        }
    }
}

/// Replaces: e464_runOn
///
/// Clones every rematerialization candidate in front of each of its uses, then erases the original
/// once nothing reads it.
///
/// ⛔⛔ `AllowOperandLiveRangeIncrease` IS A CONST GENERIC BECAUSE IT DECIDES WHICH OPS EXIST: with it
/// `false` a clone is only made where the operands are already dead, so it REMOVES clones rather than
/// being consulted, and it is what makes [`increases_operand_liverange`] reachable at all. It is a
/// `cl::opt<bool>` with `cl::init(true)` read at `:192` alone.
/// ⛔ TRAP: THE CLONES GO IN IN REVERSE PRE-ORDER, and that is what keeps the paths of the remaining
/// uses valid — inserting at a position only shifts what sits at or after it, and every earlier use is
/// either at a smaller index in that block or diverges from it at a smaller index.
/// ⚠️ TRAP: A USE THAT IMMEDIATELY FOLLOWS THE CANDIDATE IS LEFT ALONE (`:186`), so a chain of uses
/// keeps the original for its first reader and clones for the rest.
pub fn run_on<const ALLOW_OPERAND_LIVE_RANGE_INCREASE: bool>(
    unit_body: &mut Vec<Op>,
    values: &mut Values,
) {
    let candidate_results = {
        let scope: [&[Op]; 1] = [unit_body.as_slice()];
        let mut found = Vec::new();
        candidates(unit_body, Definitions::from_innermost(&scope), &mut found);
        found
    };

    for result in candidate_results {
        let to_replace = {
            let Some(cand_at) = utils::path_of(unit_body, result) else {
                continue;
            };
            let scope: [&[Op]; 1] = [unit_body.as_slice()];
            let defs = Definitions::from_innermost(&scope);
            let Some(cand_op) = cand_at.op(unit_body) else {
                continue;
            };
            let Some(cand_block) = cand_at.block(unit_body) else {
                continue;
            };
            let next = cand_at.next();
            let mut uses = Vec::new();
            uses_of(unit_body, result, &mut Vec::new(), &mut uses);
            uses.retain(|(owner_at, _)| {
                if *owner_at == next {
                    return false;
                }
                if ALLOW_OPERAND_LIVE_RANGE_INCREASE {
                    return true;
                }
                match cand_at.ancestor_in_block_of(owner_at) {
                    Some(utils::InBlock(index)) => {
                        !increases_operand_liverange(cand_op, InBlock::at(index), cand_block, defs)
                    }
                    // No ancestor in the candidate's block: the use is unreachable from it, which
                    // dominance makes impossible for a value the block defines.
                    None => false,
                }
            });
            uses
        };

        for (owner_at, operand_number) in to_replace.into_iter().rev() {
            // Re-derived per clone: the original never moves (every use follows it), and this is what
            // says so.
            let Some(original) = utils::path_of(unit_body, result)
                .and_then(|at| at.op(unit_body))
                .cloned()
            else {
                break;
            };
            let mut mapping = ValueMapping::new();
            let Some(cloned) =
                dialects::clone_ops(core::slice::from_ref(&original), values, &mut mapping).pop()
            else {
                break;
            };
            let cloned_result = dialects::results(&cloned).first().copied();
            utils::insert_at(unit_body, &owner_at, cloned);
            // `OpBuilder builder(owner)` inserted IN FRONT OF the owner, which shifted it by one.
            let owner_now = owner_at.next();
            if let (Some(new_result), Some(op)) = (cloned_result, owner_now.op_mut(unit_body)) {
                dialects::set_operand(op, operand_number, new_result);
            }
        }

        if use_count(result, unit_body) == 0 {
            dialects::erase_defining_op(unit_body, result);
        }
    }
}

/// `-dcc-rematerialization-disable`, `cl::init(false)` (`:38-40`) — a `dcc-opt` command-line flag,
/// not a program property, and this crate has no flags.
const DISABLE_THIS_PASS: bool = false;

/// `AllowOperandLiveRangeIncrease`, `cl::init(true)` (`:42-46`) — the pass entry is the one place that
/// instantiates [`run_on`]'s const generic.
const ALLOW_OPERAND_LIVE_RANGE_INCREASE: bool = true;

/// Replaces: e525_runOnOperation
///
/// The pass entry: rematerializes in every program unit of the module, unless the flag turned the pass
/// off.
///
/// ⭐ `dom_info_` IS THE DROPPABLE MECHANISM — a `DominanceInfo` built and deleted per unit is how the
/// reference reaches the answer [`InBlock`] derives from the body's own order, and `WalkResult::skip()`
/// needs no expression: a program's units are a flat list on this island.
pub fn run_on_operation<A: Arch, M: Model, W: Workload>(
    program: &mut Program<A, M, W>,
    values: &mut Values,
) {
    if DISABLE_THIS_PASS {
        return;
    }
    for unit in program.units.iter_mut() {
        run_on::<ALLOW_OPERAND_LIVE_RANGE_INCREASE>(&mut unit.body, values);
    }
}

#[cfg(test)]
mod unit_tests {
    use super::{
        Defined, InBlock, increases_operand_liverange, is_candidate_for_rematerialization,
        last_use_within_block, run_on, run_on_operation,
    };
    use crate::arch::Dd2;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::dataflow_ir::{GroupId, OpIndex, ProgramName, Units, Values};
    use crate::islands::sentient::dialects::{
        self as dialects, Definitions, Op, Val, sentient, use_count,
    };
    use crate::islands::sentient::{Program, ProgramUnit, ProgramUnits};
    use crate::model::Model;
    use crate::units::DfirUnit;
    use crate::workload::Workload;

    /// `%r = sentient.scalar_constant {value = 1 : si64} : index`.
    fn constant(result: Val) -> Op {
        Op::Sentient(sentient::Op::ScalarConstant {
            value: 1,
            result,
            reg_locale: sentient::RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    /// `%out = sentient.scalar_add %lhs, %rhs : index`.
    fn add(lhs: Val, rhs: Val, result: Val) -> Op {
        Op::Sentient(sentient::Op::ScalarAdd {
            lhs,
            rhs,
            result,
            reg: None,
            element_size: None,
            ty: ScalarTy::Index,
        })
    }

    /// `%out = sentient.scalar_mul %lhs, %rhs : index` — the op the `dyn_cast` chain never names.
    fn mul(lhs: Val, rhs: Val, result: Val) -> Op {
        Op::Sentient(sentient::Op::ScalarMul {
            lhs,
            rhs,
            result,
            reg_locale: None,
            ty: ScalarTy::Index,
        })
    }

    /// `%out = sentient.scalar_copy %input : index`.
    fn copy(input: Val, result: Val) -> Op {
        Op::Sentient(sentient::Op::ScalarCopy {
            input,
            result,
            reg: sentient::Reg {
                locale: sentient::RegType::Lrf,
                index: None,
            },
            element_size: None,
            program_header: false,
        })
    }

    /// A `sentient.for` carrying one value, running `body`.
    fn for_op(iv: Val, bound: Val, carried: sentient::Carried, body: Vec<Op>) -> Op {
        Op::Sentient(sentient::Op::For {
            iv,
            bound,
            bound_reg: None,
            carried: vec![carried],
            dbg_name: None,
            body,
        })
    }

    /// One carried value, with nothing assigned to it yet.
    fn carried(init: Val, arg: Val, result: Val) -> sentient::Carried {
        sentient::Carried {
            init,
            arg,
            result,
            reg: sentient::Reg {
                locale: sentient::RegType::Unknown,
                index: None,
            },
            program_header: false,
            element_size: None,
        }
    }

    /// `e353` — the vendor's own scenario: the operand's last use decides, and an all-constant op
    /// extends nothing to nowhere and so answers `true`.
    #[test]
    fn a_user_past_the_operands_last_use_extends_that_operands_liverange() {
        let (c1, m, a, later) = (Val(0), Val(1), Val(2), Val(3));
        let block = vec![
            constant(c1),
            mul(c1, c1, m),  // 0: the non-constant operand's definition
            add(m, c1, a),   // 1: the op under test  (block index 2)
            copy(m, later),  // 3: the operand's LAST use
            copy(a, Val(4)), // 4: the user
        ];
        let scope: [&[Op]; 1] = [block.as_slice()];
        let defs = Definitions::from_innermost(&scope);
        let op = &block[2];

        // The user is after `%1`'s last use, so cloning there keeps `%1` alive longer.
        assert!(increases_operand_liverange(
            op,
            InBlock::at(4),
            &block,
            defs
        ));
        // A user at or before that last use does not.
        assert!(!increases_operand_liverange(
            op,
            InBlock::at(3),
            &block,
            defs
        ));
        // ⛔ AN OP WITH NO NON-CONSTANT OPERAND ANSWERS `true`.
        assert!(increases_operand_liverange(
            &copy(c1, Val(5)),
            InBlock::at(0),
            &block,
            defs
        ));
    }

    /// The three candidate shapes and the three declines, including the `scalar_copy` of an iter arg
    /// that is a null deref in the reference.
    #[test]
    fn only_add_sub_and_copy_of_a_constant_are_candidates() {
        let (c1, iv, bound) = (Val(0), Val(1), Val(2));
        let body = vec![
            add(Val(10), c1, Val(11)),
            add(Val(10), Val(10), Val(12)),
            mul(Val(10), c1, Val(13)),
            copy(c1, Val(14)),
            copy(Val(10), Val(15)),
        ];
        let outer = vec![
            constant(c1),
            for_op(iv, bound, carried(c1, Val(10), Val(16)), body.clone()),
        ];
        let scopes: [&[Op]; 2] = [&body, &outer];
        let defs = Definitions::from_innermost(&scopes);
        let verdicts: Vec<bool> = body
            .iter()
            .map(|op| is_candidate_for_rematerialization(op, defs))
            .collect();
        // `%10` is the loop's iter arg, so it is a `BlockArgument` and never a constant.
        assert_eq!(verdicts, vec![true, false, false, true, false]);
    }

    /// The last use is the later position in the block, a use inside a loop body counts for the loop
    /// that encloses it, and a value read only outside the block has no last use in it.
    #[test]
    fn last_use_is_the_latest_position_that_reads_the_value() {
        let (c1, iv, bound, elsewhere) = (Val(0), Val(1), Val(2), Val(3));
        let block = vec![
            constant(c1),
            add(c1, c1, Val(11)),
            for_op(
                iv,
                bound,
                carried(Val(20), Val(21), Val(22)),
                vec![add(c1, Val(21), Val(23))],
            ),
            add(Val(11), Val(11), Val(12)),
        ];
        let scopes: [&[Op]; 1] = [&block];
        let defs = Definitions::from_innermost(&scopes);
        let c1_defined = Defined::of(c1, defs).expect("the constant binds it");
        // The `sentient.for` at position 2 is the ancestor within the block of the nested use.
        assert_eq!(
            last_use_within_block(c1_defined, &block),
            Some(InBlock::at(2))
        );
        let bound_defined = Defined::of(Val(11), defs).expect("the first add binds it");
        assert_eq!(
            last_use_within_block(bound_defined, &block),
            Some(InBlock::at(3))
        );
        // `v.use_empty()` — nothing in the block reads the loop's result.
        let result_defined = Defined::of(Val(22), defs).expect("the loop binds it");
        assert_eq!(last_use_within_block(result_defined, &block), None);
        // The `DT_CHECK_MSG` this type replaces: an iter arg has no defining op anywhere in scope.
        assert_eq!(Defined::of(Val(21), defs), None);
        assert_eq!(Defined::of(elsewhere, defs), None);
    }
    /// e464 — the second use gets a clone of its own and the first keeps the original, which therefore
    /// survives; the reader's operand is the clone's result.
    #[test]
    fn e464_run_on_clones_the_candidate_before_every_later_use() {
        let mut values = Values::default();
        for _ in 0..6 {
            let _ = values.mint();
        }
        // %0 = constant, %1 = copy %0 (the candidate), %2 = mul %1 %0, %3 = mul %1 %0. The readers
        // are `mul`s because a candidate with no reader of its own is erased below.
        let mut body = vec![
            constant(Val(0)),
            copy(Val(0), Val(1)),
            mul(Val(1), Val(0), Val(2)),
            mul(Val(1), Val(0), Val(3)),
        ];
        run_on::<true>(&mut body, &mut values);

        assert_eq!(body.len(), 5);
        // The immediately following use kept the original copy.
        assert_eq!(dialects::operands(&body[2]), vec![Val(1), Val(0)]);
        // The later use reads the clone that now sits in front of it.
        let cloned = dialects::results(&body[3])[0];
        assert!(matches!(
            &body[3],
            Op::Sentient(sentient::Op::ScalarCopy { .. })
        ));
        assert_eq!(dialects::operands(&body[4]), vec![cloned, Val(0)]);
    }

    /// e464 — the original is erased once every use has taken a clone.
    #[test]
    fn e464_run_on_erases_a_candidate_nothing_reads_any_more() {
        let mut values = Values::default();
        for _ in 0..6 {
            let _ = values.mint();
        }
        // The one use does not immediately follow the candidate, so it is rematerialized and the
        // original is left with no reader.
        let mut body = vec![
            constant(Val(0)),
            copy(Val(0), Val(1)),
            constant(Val(4)),
            mul(Val(1), Val(4), Val(2)),
        ];
        run_on::<true>(&mut body, &mut values);

        assert_eq!(body.len(), 4);
        assert!(matches!(
            &body[0],
            Op::Sentient(sentient::Op::ScalarConstant { .. })
        ));
        assert!(matches!(
            &body[1],
            Op::Sentient(sentient::Op::ScalarConstant { .. })
        ));
        let cloned = dialects::results(&body[2])[0];
        assert_ne!(cloned, Val(1));
        assert_eq!(dialects::operands(&body[3]), vec![cloned, Val(4)]);
        assert_eq!(use_count(Val(1), &body), 0);
    }
    /// A model and a rung, so the program is typed; nothing here reads either.
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

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct AnyRung;
    impl Workload for AnyRung {
        const ROWS: u32 = 1;
        const ACTIVE_CAP: u32 = 64;
    }

    /// A two-unit program, both units running `body`.
    fn two_units(body: Vec<Op>) -> Program<Dd2, AnyModel, AnyRung> {
        let unit = |body| ProgramUnit {
            on: Units::one(DfirUnit::Lxlu, Val(0)),
            precision: None,
            body,
            arch: core::marker::PhantomData,
        };
        Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            preamble: Vec::new(),
            units: ProgramUnits::of(unit(body.clone()), vec![unit(body)]),
            bound: core::marker::PhantomData,
        }
    }

    /// e525 — the pass entry rematerializes in EVERY program unit, which is all it adds over e464.
    #[test]
    fn e525_rematerializes_in_every_unit_of_the_program() {
        let mut values = Values::default();
        for _ in 0..6 {
            let _ = values.mint();
        }
        let mut program = two_units(vec![
            constant(Val(0)),
            copy(Val(0), Val(1)),
            constant(Val(4)),
            mul(Val(1), Val(4), Val(2)),
        ]);

        run_on_operation(&mut program, &mut values);

        for unit in program.units.iter() {
            // The candidate was cloned in front of its one use and the original erased.
            assert_eq!(use_count(Val(1), &unit.body), 0);
            assert!(matches!(
                &unit.body[2],
                Op::Sentient(sentient::Op::ScalarCopy { .. })
            ));
        }
    }
}
