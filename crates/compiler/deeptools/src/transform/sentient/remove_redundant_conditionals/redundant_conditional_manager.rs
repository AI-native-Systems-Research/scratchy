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

//! `RemoveRedundantConditionals.cpp` — 4 of the campaign's 656 units (dependency level(s) [1, 2, 3]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e354_getReturnValsIfSimpleConditional` | 354 | 1 | 27 | `dcc/src/Transform/Sentient/RemoveRedundantConditionals.cpp:121` |
//! | `e465_updateIfOpBasedOnParentIfOp` | 465 | 2 | 65 | `dcc/src/Transform/Sentient/RemoveRedundantConditionals.cpp:151` |
//! | `e466_updateIfOpFeedingDynLoopBound` | 466 | 2 | 77 | `dcc/src/Transform/Sentient/RemoveRedundantConditionals.cpp:217` |
//! | `e526_processIfOp` | 526 | 3 | 22 | `dcc/src/Transform/Sentient/RemoveRedundantConditionals.cpp:98` |

// ⭐ `e575_runOnOperation` HAS LANDED and calls [`process_if_op`], so this file's own note is
// discharged; the `allow` it asked for now sits on the parent module, where the ONE remaining
// unreached root — the pass entry itself — is.

use crate::bridges::dataflow_ir_to_sentient::tf_cfgs_dataflow_conditional_tree::{
    DbgNamePrefix, new_dbg_name_from_list,
};
use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::dialects::{
    self as dialects, Definitions, Op, Val, sentient, use_count,
};
use crate::transform::sentient::ForRef;
use crate::transform::sentient::utils::{self, InBlock, OpAt};

/// `typedef int64_t WidestIntType` (`:52`) — the width the pass compares yielded constants at.
///
/// ⭐ AN ALIAS AND NOT A NEWTYPE: [`sentient::Op::ScalarConstant::value`] is already the `i64` this
/// names, and wrapping it here would only be unwrapped again at the one comparison it exists for.
pub type WidestInt = i64;

/// A `sentient.if` — `sentient::IfOp if_op` as a witness over the union type, so that the two regions
/// this reads are reachable without a second `dyn_cast`.
#[derive(Debug, Clone, Copy)]
pub struct IfOp<'a> {
    /// `if_op.getThenRegion().front()`.
    then_body: &'a [Op],
    /// `if_op.getElseRegion().front()`.
    else_body: &'a [Op],
}

impl<'a> IfOp<'a> {
    /// The witness, or `None` for any other op.
    #[must_use]
    pub fn of(op: &'a Op) -> Option<IfOp<'a>> {
        match op {
            Op::Sentient(sentient::Op::If {
                then_body,
                else_body,
                ..
            }) => Some(IfOp {
                then_body,
                else_body,
            }),
            _ => None,
        }
    }
}

/// Replaces: e354_getReturnValsIfSimpleConditional
///
/// The `(then-val, else-val)` constants yielded at `idx` when BOTH regions hold nothing but their
/// yield, and `None` otherwise (`:121-148`).
///
/// ⛔ `None` COVERS ALL THREE REFUSALS: a region with more than the yield in it, and either side
/// yielding something that is not a `sentient.scalar_constant`.
/// ⛔ TRAP: THE CONSTANTS ARE DEFINED OUTSIDE THE REGIONS, which is why `defs` is a parameter — the
/// yield's operand is looked up in the enclosing scopes, innermost first.
/// ⛔ `dyn_cast_or_null` (`:130`, `:139`) IS ALREADY TOTAL HERE: a yielded region argument has no
/// defining op, and [`Definitions::of`] answers `None` where the reference relies on the null-tolerant
/// spelling to avoid an assert.
#[must_use]
pub fn get_return_vals_if_simple_conditional(
    if_op: IfOp<'_>,
    idx: usize,
    defs: Definitions<'_>,
) -> Option<(WidestInt, WidestInt)> {
    let then_val = yielded_constant(if_op.then_body, idx, defs)?;
    let else_val = yielded_constant(if_op.else_body, idx, defs)?;
    Some((then_val, else_val))
}

/// One region's `front().getOperations().size() != 1` check and its terminator's `idx`th operand.
///
/// ⭐ THE TERMINATOR IS THE ONE OP, so the size check and the `getTerminator()` are the same match.
fn yielded_constant(body: &[Op], idx: usize, defs: Definitions<'_>) -> Option<WidestInt> {
    let [Op::Sentient(sentient::Op::Yield { results })] = body else {
        return None;
    };
    match defs.of(*results.get(idx)?) {
        Some(Op::Sentient(sentient::Op::ScalarConstant { value, .. })) => Some(*value),
        _ => None,
    }
}

/// ONE OP THE MANAGER QUEUED FOR `erase()` — `llvm::SmallVector<Operation *, 4>& to_be_deleted_`.
///
/// ⛔⛔ AN IDENTITY AND NEVER A PATH: the queue is drained only after the WHOLE conditional-tree walk
/// (`:319`), by which time every position in it has moved. A [`Val`] is minted once, so it still names
/// the op that binds it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Doomed {
    /// A `sentient.if`, named by the first result it binds.
    If(Val),
    /// A `sentient.for`, which binds nothing here — named by its induction variable ([`ForRef`]).
    For(ForRef),
}

/// HOW MANY OPS A REWRITE PUT DIRECTLY AFTER THE CONDITIONAL IT REWROTE — 0, or e466's two.
///
/// ⛔⛔ THE REFERENCE NEEDS NO SUCH ANSWER: its tree node holds an `Operation *`, which an insertion
/// into the block cannot move, while a path is an index. The walk that carries one has to step over
/// what landed behind it, and over the conditional e466 created, which is no node of that tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InsertedAfter(pub usize);

/// WHAT THE PARENT CONDITIONAL LETS THIS ONE DO — the two arms of `:172` and `:194` as one answer, so
/// that every read of the IR happens before the first write to it.
#[derive(Debug, Clone)]
enum ParentFold {
    /// The parent yields the same value in both branches: its result is that value.
    SameBothBranches {
        /// `parent_if_op.getResult(result_idx)`.
        parent_result: Val,
        /// The operand the then-region yields there — a constant defined OUTSIDE both regions.
        value: Val,
    },
    /// The child's condition becomes the parent's, or its negation.
    TakeParentPredicate {
        /// `parent_if_op.getPredicate()`, negated when the child matched the ELSE constant.
        predicate: sentient::CmpPredicate,
        /// `parent_if_op.getOperand(0)`.
        lhs: Val,
        /// `parent_if_op.getOperand(1)`.
        rhs: Val,
        /// The `RRC(parent, child)` name, or `None` when either op has none.
        dbg_name: Option<String>,
    },
}

/// The fold plus the parent results `parent_if_op.use_empty()` is asked of.
#[derive(Debug, Clone)]
struct Fold {
    action: ParentFold,
    parent_results: Vec<Val>,
}

/// Everything e465 reads before it writes anything — `None` at each of the reference's five refusals.
fn parent_fold(
    unit_body: &[Op],
    if_at: &OpAt,
    non_const_side: Val,
    const_val: WidestInt,
) -> Option<Fold> {
    let scope: [&[Op]; 1] = [unit_body];
    let defs = Definitions::from_innermost(&scope);
    let Op::Sentient(sentient::Op::If {
        predicate,
        lhs,
        rhs,
        dbg_name: child_dbg_name,
        ..
    }) = if_at.op(unit_body)?
    else {
        return None;
    };
    // `isa<BlockArgument>` on either side of the child's predicate — a region argument is bound by no
    // op — and the `eq`-only guard.
    if defs.of(*lhs).is_none()
        || defs.of(*rhs).is_none()
        || *predicate != sentient::CmpPredicate::Eq
    {
        return None;
    }
    let parent = defs.of(non_const_side)?;
    let Op::Sentient(sentient::Op::If {
        predicate: parent_predicate,
        lhs: parent_lhs,
        rhs: parent_rhs,
        yielded,
        dbg_name: parent_dbg_name,
        then_body,
        ..
    }) = parent
    else {
        return None;
    };
    let result_idx = yielded
        .iter()
        .position(|value| value.result == non_const_side)?;
    let parent_results: Vec<Val> = yielded.iter().map(|value| value.result).collect();
    let (then_val, else_val) =
        get_return_vals_if_simple_conditional(IfOp::of(parent)?, result_idx, defs)?;
    if then_val == else_val {
        let Some(Op::Sentient(sentient::Op::Yield { results })) = then_body.first() else {
            return None;
        };
        return Some(Fold {
            action: ParentFold::SameBothBranches {
                parent_result: non_const_side,
                value: *results.get(result_idx)?,
            },
            parent_results,
        });
    }
    if const_val != then_val && const_val != else_val {
        return None;
    }
    let negate_parent_predicate = const_val == else_val;
    Some(Fold {
        action: ParentFold::TakeParentPredicate {
            predicate: if negate_parent_predicate {
                utils::negate_predicate(*parent_predicate)
            } else {
                *parent_predicate
            },
            lhs: *parent_lhs,
            rhs: *parent_rhs,
            dbg_name: new_dbg_name_from_list(
                DbgNamePrefix::Rrc,
                parent_dbg_name.as_deref(),
                &[child_dbg_name.as_deref()],
            ),
        },
        parent_results,
    })
}

/// Replaces: e465_updateIfOpBasedOnParentIfOp
///
/// Gives this conditional its parent's condition (negated when it tested the parent's ELSE constant),
/// or folds a parent that yields one value in both branches away, and queues the parent once dead.
///
/// ⛔ TRAP: A `None` DEBUG NAME LEAVES THE CHILD'S OWN NAME ALONE — `setDbgNameAttr(op, nullptr)`
/// would REMOVE the attribute, which is why the reference guards the call with `if (StringAttr ..)`.
/// ⭐ `OpBuilder builder(if_op_)` (`:155`) IS DEAD IN THE REFERENCE: nothing is created here.
pub fn update_if_op_based_on_parent_if_op(
    unit_body: &mut Vec<Op>,
    if_at: &OpAt,
    non_const_side: Val,
    const_val: WidestInt,
    to_be_deleted: &mut Vec<Doomed>,
) {
    let Some(fold) = parent_fold(unit_body, if_at, non_const_side, const_val) else {
        return;
    };
    match fold.action {
        ParentFold::SameBothBranches {
            parent_result,
            value,
        } => dialects::replace_all_uses_with(unit_body, parent_result, value),
        ParentFold::TakeParentPredicate {
            predicate: new_predicate,
            lhs: new_lhs,
            rhs: new_rhs,
            dbg_name: new_dbg_name,
        } => {
            if let Some(Op::Sentient(sentient::Op::If {
                predicate,
                lhs,
                rhs,
                dbg_name,
                ..
            })) = if_at.op_mut(unit_body)
            {
                *predicate = new_predicate;
                *lhs = new_lhs;
                *rhs = new_rhs;
                if let Some(name) = new_dbg_name {
                    *dbg_name = Some(name);
                }
            }
        }
    }
    if fold
        .parent_results
        .iter()
        .all(|val| use_count(*val, unit_body) == 0)
    {
        if let Some(&first) = fold.parent_results.first() {
            to_be_deleted.push(Doomed::If(first));
        }
    }
}

/// Everything e466 reads before it writes anything — the conditional, its one loop and the bound.
#[derive(Debug, Clone)]
struct FuseLoop {
    /// `if_op_.getResult(0)`.
    if_result: Val,
    /// Where the loop sits in the conditional's own block.
    for_index: InBlock,
    /// The loop's induction variable — its identity once it is queued.
    iv: Val,
    /// The new conditional's predicate, negated when the then-branch was the zero one.
    predicate: sentient::CmpPredicate,
    /// `if_op_.getLhs()`.
    lhs: Val,
    /// `if_op_.getRhs()`.
    rhs: Val,
    /// `getDbgNameAttr(if_op_)`, copied onto the new conditional.
    dbg_name: Option<String>,
    /// `nonzero_ret_val`.
    bound: WidestInt,
}

/// The four refusals of `:219-232` as one `None`.
fn fuse_plan(unit_body: &[Op], if_at: &OpAt) -> Option<FuseLoop> {
    let scope: [&[Op]; 1] = [unit_body];
    let defs = Definitions::from_innermost(&scope);
    let if_op = if_at.op(unit_body)?;
    let Op::Sentient(sentient::Op::If {
        predicate,
        lhs,
        rhs,
        yielded,
        dbg_name,
        ..
    }) = if_op
    else {
        return None;
    };
    let [only] = yielded.as_slice() else {
        return None;
    };
    if use_count(only.result, unit_body) != 1 {
        return None;
    }
    // The one user, a result-less `sentient.for` bound by it, IN THE CONDITIONAL'S OWN BLOCK — a use
    // anywhere else is a user whose `getBlock()` differs, which the reference declines.
    let (for_index, iv) = if_at
        .block(unit_body)?
        .iter()
        .enumerate()
        .find_map(|(index, op)| match op {
            Op::Sentient(sentient::Op::For {
                iv, bound, carried, ..
            }) if *bound == only.result && carried.is_empty() => Some((InBlock(index), *iv)),
            _ => None,
        })?;
    let (then_val, else_val) = get_return_vals_if_simple_conditional(IfOp::of(if_op)?, 0, defs)?;
    if then_val != 0 && else_val != 0 {
        return None;
    }
    // `DT_CHECK_MSG(first != second, "Expect conditionals yielding the same value in both branches to
    // have been cleaned up")` — with one side zero, equal means both are, and e465's own fold is what
    // removes that conditional. Nothing to fuse rather than an abort.
    if then_val == else_val {
        return None;
    }
    let to_negate_predicate = then_val == 0;
    Some(FuseLoop {
        if_result: only.result,
        for_index,
        iv,
        predicate: if to_negate_predicate {
            utils::negate_predicate(*predicate)
        } else {
            *predicate
        },
        lhs: *lhs,
        rhs: *rhs,
        dbg_name: dbg_name.clone(),
        bound: if to_negate_predicate {
            else_val
        } else {
            then_val
        },
    })
}

/// Replaces: e466_updateIfOpFeedingDynLoopBound
///
/// Fuses a loop whose dynamic bound is this conditional into the branch that gives it a NON-ZERO
/// bound: a fresh result-less conditional after it takes the loop (or, at a bound of one, the loop's
/// body), the loop's bound becomes that constant, and both old ops are queued.
///
/// ⭐ AN EMPTY `else_body` IS THE REFERENCE'S ELSE REGION WITH NO BLOCK IN IT — see
/// [`sentient::Op::If`], where empty means there is no `else` at all.
/// ⚠️ TRAP: THE LOOP SITS TWO POSITIONS LOWER once the constant and the new conditional are in, which
/// is where the reference's `Operation *` needs no adjustment and a path does.
/// ⚠️ TRAP: AT A BOUND OF ONE THE LOOP'S OWN BODY IS MOVED OUT of it, up to but not including its
/// yield, and the emptied loop is queued — its induction variable goes with it.
pub fn update_if_op_feeding_dyn_loop_bound(
    unit_body: &mut Vec<Op>,
    if_at: &OpAt,
    to_be_deleted: &mut Vec<Doomed>,
    values: &mut Values,
) -> InsertedAfter {
    let Some(plan) = fuse_plan(unit_body, if_at) else {
        return InsertedAfter(0);
    };
    let new_loop_bound = values.mint();
    let for_at = if_at.sibling(plan.for_index);
    if let Some(Op::Sentient(sentient::Op::For { bound, .. })) = for_at.op_mut(unit_body) {
        *bound = new_loop_bound;
    }
    // `setInsertionPointAfter(if_op_)`, then the constant and the conditional in creation order.
    let constant_at = if_at.next();
    utils::insert_at(
        unit_body,
        &constant_at,
        Op::Sentient(sentient::Op::ScalarConstant {
            value: plan.bound,
            result: new_loop_bound,
            // `ConstantOp`'s own default, which this creation does not override, and a loop bound is
            // an `index`.
            reg_locale: sentient::RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        }),
    );
    let new_if_at = constant_at.next();
    utils::insert_at(
        unit_body,
        &new_if_at,
        Op::Sentient(sentient::Op::If {
            predicate: plan.predicate,
            lhs: plan.lhs,
            rhs: plan.rhs,
            // No results, so no `regLocales` either.
            yielded: Vec::new(),
            dbg_name: plan.dbg_name,
            then_body: vec![Op::Sentient(sentient::Op::Yield {
                results: Vec::new(),
            })],
            else_body: Vec::new(),
        }),
    );
    let for_now = if_at.sibling(InBlock(plan.for_index.0 + 2));
    let moved = if plan.bound == 1 {
        to_be_deleted.push(Doomed::For(ForRef(plan.iv)));
        match for_now.op_mut(unit_body) {
            Some(Op::Sentient(sentient::Op::For { body, .. })) => {
                let upto = body
                    .iter()
                    .position(|op| matches!(op, Op::Sentient(sentient::Op::Yield { .. })))
                    .unwrap_or(body.len());
                body.drain(..upto).collect()
            }
            _ => Vec::new(),
        }
    } else {
        utils::remove_at(unit_body, &for_now).map_or_else(Vec::new, |op| vec![op])
    };
    if let Some(Op::Sentient(sentient::Op::If { then_body, .. })) = new_if_at.op_mut(unit_body) {
        then_body.splice(0..0, moved);
    }
    to_be_deleted.push(Doomed::If(plan.if_result));
    InsertedAfter(2)
}

/// Which side of the predicate a `sentient.scalar_constant` defines — `const_val_` with the OTHER side
/// as `non_const_side_`.
fn predicate_constant(unit_body: &[Op], if_at: &OpAt) -> Option<(WidestInt, Val)> {
    let scope: [&[Op]; 1] = [unit_body];
    let defs = Definitions::from_innermost(&scope);
    let Op::Sentient(sentient::Op::If { lhs, rhs, .. }) = if_at.op(unit_body)? else {
        return None;
    };
    let constant = |val: Val| match defs.of(val) {
        Some(Op::Sentient(sentient::Op::ScalarConstant { value, .. })) => Some(*value),
        _ => None,
    };
    match (constant(*lhs), constant(*rhs)) {
        (Some(value), _) => Some((value, *rhs)),
        (None, Some(value)) => Some((value, *lhs)),
        (None, None) => None,
    }
}

/// Replaces: e526_processIfOp
///
/// Reads the constant side of one conditional's predicate and runs both rewrites off it; a predicate
/// comparing two non-constants is left alone.
///
/// ⛔ TRAP: `isa<BlockArgument>` IS A NULL GUARD, NOT A CASE — a bare `dyn_cast` on a region argument's
/// null `getDefiningOp()` asserts, and [`Definitions::of`] answering [`None`] covers both.
/// ⚠️ TRAP: THE LHS WINS WHEN BOTH SIDES ARE CONSTANT, so `non_const_side_` is then the constant RHS.
/// ⭐ IT ANSWERS [`InsertedAfter`], which is the walk's business and none of the reference's.
pub fn process_if_op(
    unit_body: &mut Vec<Op>,
    if_at: &OpAt,
    to_be_deleted: &mut Vec<Doomed>,
    values: &mut Values,
) -> InsertedAfter {
    let Some((const_val, non_const_side)) = predicate_constant(unit_body, if_at) else {
        return InsertedAfter(0);
    };
    update_if_op_based_on_parent_if_op(unit_body, if_at, non_const_side, const_val, to_be_deleted);
    update_if_op_feeding_dyn_loop_bound(unit_body, if_at, to_be_deleted, values)
}

#[cfg(test)]
mod unit_tests {
    use super::{
        Doomed, IfOp, get_return_vals_if_simple_conditional, process_if_op,
        update_if_op_based_on_parent_if_op, update_if_op_feeding_dyn_loop_bound,
    };
    use crate::islands::dataflow_ir::Values;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::{Definitions, Op, Val, sentient};
    use crate::transform::sentient::ForRef;
    use crate::transform::sentient::utils::{InBlock, OpAt};

    /// `%r = sentient.scalar_constant {value = <value>} : index`.
    fn constant(result: Val, value: i64) -> Op {
        Op::Sentient(sentient::Op::ScalarConstant {
            value,
            result,
            reg_locale: sentient::RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    /// `sentient.yield %results`.
    fn yield_op(results: Vec<Val>) -> Op {
        Op::Sentient(sentient::Op::Yield { results })
    }

    /// `%r = sentient.if eq(%lhs, %rhs) { <then> } else { <else> }`.
    fn if_op(result: Val, then_body: Vec<Op>, else_body: Vec<Op>) -> Op {
        Op::Sentient(sentient::Op::If {
            predicate: sentient::CmpPredicate::Eq,
            lhs: Val(0),
            rhs: Val(1),
            yielded: vec![sentient::Yielded {
                result,
                reg: sentient::Reg {
                    locale: sentient::RegType::Unknown,
                    index: None,
                },
                element_size: None,
            }],
            dbg_name: None,
            then_body,
            else_body,
        })
    }

    /// `e354` — the simple conditional's pair, and each of the three refusals.
    #[test]
    fn a_pair_of_constants_needs_both_regions_to_hold_only_the_yield() {
        let (c3, c7, other) = (Val(2), Val(3), Val(4));
        let simple = if_op(Val(5), vec![yield_op(vec![c3])], vec![yield_op(vec![c7])]);
        let body = vec![constant(c3, 3), constant(c7, 7), simple];
        let scope: [&[Op]; 1] = [body.as_slice()];
        let defs = Definitions::from_innermost(&scope);
        let witness = IfOp::of(&body[2]).expect("a sentient.if");
        assert_eq!(
            get_return_vals_if_simple_conditional(witness, 0, defs),
            Some((3, 7))
        );
        // ⛔ NO SUCH INDEX.
        assert_eq!(
            get_return_vals_if_simple_conditional(witness, 1, defs),
            None
        );
        // ⛔ A REGION WITH MORE THAN ITS YIELD IN IT.
        let busy = if_op(
            Val(6),
            vec![constant(other, 9), yield_op(vec![other])],
            vec![yield_op(vec![c7])],
        );
        assert_eq!(
            get_return_vals_if_simple_conditional(IfOp::of(&busy).unwrap(), 0, defs),
            None
        );
        // ⛔ A YIELDED VALUE THAT IS NOT A CONSTANT — here nothing in scope defines it.
        let unknown = if_op(
            Val(7),
            vec![yield_op(vec![Val(99)])],
            vec![yield_op(vec![c7])],
        );
        assert_eq!(
            get_return_vals_if_simple_conditional(IfOp::of(&unknown).unwrap(), 0, defs),
            None
        );
        // ⛔ AND ONLY A `sentient.if` IS A WITNESS.
        assert!(IfOp::of(&body[0]).is_none());
    }

    /// `%r = sentient.if <predicate>(%lhs, %rhs) { <then> } else { <else> }`, named.
    fn cond(
        result: Val,
        predicate: sentient::CmpPredicate,
        (lhs, rhs): (Val, Val),
        dbg_name: Option<&str>,
        then_body: Vec<Op>,
        else_body: Vec<Op>,
    ) -> Op {
        Op::Sentient(sentient::Op::If {
            predicate,
            lhs,
            rhs,
            yielded: vec![sentient::Yielded {
                result,
                reg: sentient::Reg {
                    locale: sentient::RegType::Unknown,
                    index: None,
                },
                element_size: None,
            }],
            dbg_name: dbg_name.map(String::from),
            then_body,
            else_body,
        })
    }

    /// `sentient.for %iv = %bound { <body> }`, carrying nothing.
    fn for_op(iv: Val, bound: Val, body: Vec<Op>) -> Op {
        Op::Sentient(sentient::Op::For {
            iv,
            bound,
            bound_reg: None,
            carried: Vec::new(),
            dbg_name: None,
            body,
        })
    }

    /// `e465` — the child takes the NEGATION of its parent's predicate because it tested the ELSE
    /// constant, gains the parent's operands and the `RRC(..)` name, and the dead parent is queued.
    #[test]
    fn a_child_conditional_takes_over_the_predicate_that_decided_its_operand() {
        let (a, b, c5, c9) = (Val(0), Val(1), Val(2), Val(3));
        let (parent_result, child_result) = (Val(4), Val(5));
        let mut body = vec![
            constant(a, 3),
            constant(b, 4),
            constant(c5, 5),
            constant(c9, 9),
            cond(
                parent_result,
                sentient::CmpPredicate::Slt,
                (a, b),
                Some("p"),
                vec![yield_op(vec![c5])],
                vec![yield_op(vec![c9])],
            ),
            cond(
                child_result,
                sentient::CmpPredicate::Eq,
                (parent_result, c9),
                Some("c"),
                vec![yield_op(vec![c5])],
                vec![yield_op(vec![c9])],
            ),
        ];
        let child_at = OpAt::top(InBlock(5));
        let mut to_be_deleted = Vec::new();
        update_if_op_based_on_parent_if_op(
            &mut body,
            &child_at,
            parent_result,
            9,
            &mut to_be_deleted,
        );
        let Op::Sentient(sentient::Op::If {
            predicate,
            lhs,
            rhs,
            dbg_name,
            ..
        }) = &body[5]
        else {
            panic!("the child is still a sentient.if");
        };
        assert_eq!(*predicate, sentient::CmpPredicate::Sge);
        assert_eq!((*lhs, *rhs), (a, b));
        assert_eq!(dbg_name.as_deref(), Some("RRC(p, c)"));
        assert_eq!(to_be_deleted, vec![Doomed::If(parent_result)]);
    }

    /// `e466` — the loop moves into the then-branch of a fresh result-less conditional whose
    /// predicate is negated (the then-branch yielded the zero bound), reading the new constant.
    #[test]
    fn a_loop_bound_by_a_conditional_moves_into_its_nonzero_branch() {
        let (a, b, zero, four) = (Val(0), Val(1), Val(2), Val(3));
        let (if_result, iv) = (Val(4), Val(5));
        let mut body = vec![
            constant(a, 3),
            constant(b, 4),
            constant(zero, 0),
            constant(four, 4),
            cond(
                if_result,
                sentient::CmpPredicate::Slt,
                (a, b),
                Some("dyn"),
                vec![yield_op(vec![zero])],
                vec![yield_op(vec![four])],
            ),
            for_op(iv, if_result, vec![constant(Val(6), 1), yield_op(vec![])]),
        ];
        let mut values = Values::default();
        // Val(0)..=Val(6) are taken above.
        for _ in 0..7 {
            let _ = values.mint();
        }
        let mut to_be_deleted = Vec::new();
        update_if_op_feeding_dyn_loop_bound(
            &mut body,
            &OpAt::top(InBlock(4)),
            &mut to_be_deleted,
            &mut values,
        );
        assert_eq!(body.len(), 7);
        let Op::Sentient(sentient::Op::ScalarConstant { value, result, .. }) = &body[5] else {
            panic!("the non-zero bound is a fresh constant right after the old conditional");
        };
        let (bound_value, new_bound) = (*value, *result);
        assert_eq!(bound_value, 4);
        let Op::Sentient(sentient::Op::If {
            predicate,
            lhs,
            rhs,
            yielded,
            dbg_name,
            then_body,
            else_body,
        }) = &body[6]
        else {
            panic!("the new conditional follows the constant");
        };
        assert_eq!(*predicate, sentient::CmpPredicate::Sge);
        assert_eq!((*lhs, *rhs), (a, b));
        assert!(yielded.is_empty() && else_body.is_empty());
        assert_eq!(dbg_name.as_deref(), Some("dyn"));
        assert_eq!(
            then_body,
            &vec![
                for_op(iv, new_bound, vec![constant(Val(6), 1), yield_op(vec![])]),
                yield_op(vec![]),
            ]
        );
        assert_eq!(to_be_deleted, vec![Doomed::If(if_result)]);
    }

    /// `e466` — at a bound of ONE the loop's own body is what moves, and the emptied loop is queued.
    #[test]
    fn a_trip_count_of_one_moves_the_loop_body_and_queues_the_loop() {
        let (a, b, zero, one) = (Val(0), Val(1), Val(2), Val(3));
        let (if_result, iv, inner) = (Val(4), Val(5), Val(6));
        let mut body = vec![
            constant(a, 3),
            constant(b, 4),
            constant(zero, 0),
            constant(one, 1),
            cond(
                if_result,
                sentient::CmpPredicate::Slt,
                (a, b),
                None,
                vec![yield_op(vec![one])],
                vec![yield_op(vec![zero])],
            ),
            for_op(iv, if_result, vec![constant(inner, 7), yield_op(vec![])]),
        ];
        let mut values = Values::default();
        // Val(0)..=Val(6) are taken above.
        for _ in 0..7 {
            let _ = values.mint();
        }
        let mut to_be_deleted = Vec::new();
        update_if_op_feeding_dyn_loop_bound(
            &mut body,
            &OpAt::top(InBlock(4)),
            &mut to_be_deleted,
            &mut values,
        );
        // ⭐ THE THEN-BRANCH KEPT ITS PREDICATE: the then-branch already yielded the non-zero bound.
        let Op::Sentient(sentient::Op::If {
            predicate,
            then_body,
            ..
        }) = &body[6]
        else {
            panic!("the new conditional follows the constant");
        };
        assert_eq!(*predicate, sentient::CmpPredicate::Slt);
        assert_eq!(then_body, &vec![constant(inner, 7), yield_op(vec![])]);
        // The loop stays where it is, emptied, until the queue is drained.
        assert_eq!(body[7], for_op(iv, Val(7), vec![yield_op(vec![])]));
        assert_eq!(
            to_be_deleted,
            vec![Doomed::For(ForRef(iv)), Doomed::If(if_result)]
        );
    }
    /// e526 — both rewrites run off the ONE constant the predicate names, and a predicate with no
    /// constant side is left alone.
    #[test]
    fn process_if_op_needs_a_constant_side_before_either_rewrite_runs() {
        let (a, b, zero, four) = (Val(0), Val(1), Val(2), Val(3));
        let (if_result, iv) = (Val(4), Val(5));
        let mut body = vec![
            constant(a, 3),
            constant(b, 4),
            constant(zero, 0),
            constant(four, 4),
            cond(
                if_result,
                sentient::CmpPredicate::Slt,
                (a, b),
                Some("dyn"),
                vec![yield_op(vec![zero])],
                vec![yield_op(vec![four])],
            ),
            for_op(iv, if_result, vec![constant(Val(6), 1), yield_op(vec![])]),
        ];
        let mut values = Values::default();
        for _ in 0..7 {
            let _ = values.mint();
        }
        let mut to_be_deleted = Vec::new();

        process_if_op(
            &mut body,
            &OpAt::top(InBlock(4)),
            &mut to_be_deleted,
            &mut values,
        );

        // ⚠️ THE LHS WON, so `non_const_side_` was the constant `b` — and with no parent conditional
        // the first rewrite found nothing, leaving the dynamic-bound one to move the loop.
        assert_eq!(body.len(), 7);
        assert!(matches!(
            &body[6],
            Op::Sentient(sentient::Op::If { then_body, .. }) if then_body.len() == 2
        ));
        assert_eq!(to_be_deleted, vec![Doomed::If(if_result)]);

        // ⛔ AND WITH NEITHER SIDE CONSTANT, NOTHING AT ALL.
        let mut untouched = vec![
            constant(zero, 0),
            constant(four, 4),
            cond(
                if_result,
                sentient::CmpPredicate::Slt,
                (Val(7), Val(8)),
                Some("dyn"),
                vec![yield_op(vec![zero])],
                vec![yield_op(vec![four])],
            ),
            for_op(iv, if_result, vec![yield_op(vec![])]),
        ];
        let before = untouched.clone();
        let mut none_deleted = Vec::new();
        process_if_op(
            &mut untouched,
            &OpAt::top(InBlock(2)),
            &mut none_deleted,
            &mut values,
        );
        assert_eq!(untouched, before);
        assert!(none_deleted.is_empty());
    }
}
