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

//! `RemoveRedundantConditionals.cpp` — 1 of the campaign's 656 units (dependency level(s) [4]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e575_runOnOperation` | 575 | 4 | 28 | `dcc/src/Transform/Sentient/RemoveRedundantConditionals.cpp:295` |

// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET: `e575_runOnOperation` below is this module's only
// root and nothing calls it, so every item under it — this file's and the manager's — reads as dead.
// CI runs clippy with `-D warnings`, so without this the pass entry itself fails the gate.
// ⭐ REMOVE THIS WHEN THE PIPELINE CALLS THE PASS: at that point an unused item here is a real defect.
#![allow(dead_code)]

pub(crate) mod redundant_conditional_manager;

use crate::arch::Arch;
use crate::islands::dataflow_ir::Values;
use crate::islands::sentient::Program;
use crate::islands::sentient::dialects::{
    Op, Val, erase_defining_op, regions_mut, regions_ref, sentient,
};
use crate::model::Model;
use crate::transform::sentient::ForRef;
use crate::transform::sentient::remove_redundant_conditionals::redundant_conditional_manager::{
    Doomed, process_if_op,
};
use crate::transform::sentient::utils::{InBlock, OpAt};
use crate::workload::Workload;

/// `-dcc-remove-redundant-conditionals-disable`, `cl::init(false)` (`:45-48`) — a `dcc-opt` flag, not
/// a program property, and this crate has no flags.
const DISABLE_THIS_PASS: bool = false;

/// `dcc::ConditionalTree`'s nodes that carry a `sentient.if` — the tree itself is the droppable
/// mechanism for reaching them, so what is left is where each one SITS.
fn if_paths(scope: &[Op], enclosing: &mut Vec<(InBlock, usize)>, out: &mut Vec<OpAt>) {
    for (index, op) in scope.iter().enumerate() {
        if matches!(op, Op::Sentient(sentient::Op::If { .. })) {
            out.push(OpAt::at(enclosing, InBlock(index)));
        }
        for (region, block) in regions_ref(op).into_iter().enumerate() {
            enclosing.push((InBlock(index), region));
            if_paths(block, enclosing, out);
            enclosing.pop();
        }
    }
}

/// `Operation::erase()` FOR AN OP THAT BINDS NOTHING: a `sentient.for`'s results are its carried
/// values, never its induction variable, so [`erase_defining_op`] cannot reach one queued by e466.
fn erase_for_op(scope: &mut Vec<Op>, iv: Val) {
    let is_it = |op: &Op| matches!(op, Op::Sentient(sentient::Op::For { iv: at, .. }) if *at == iv);
    if let Some(at) = scope.iter().position(is_it) {
        scope.remove(at);
        return;
    }
    for op in scope.iter_mut() {
        for region in regions_mut(op) {
            erase_for_op(region, iv);
        }
    }
}

/// Replaces: e575_runOnOperation
///
/// The pass entry: simplifies every conditional of every program unit, then erases the ops the
/// simplifications emptied — all of them, and only once the whole unit has been walked (`:295-322`).
///
/// ⚠️ TRAP: THE QUEUE IS DRAINED PER UNIT AND NOT PER CONDITIONAL, because e466 leaves an emptied
/// `sentient.for` standing while a later conditional may still be read against it.
/// ⚠️ DIVERGENCE: `CondNode::walk<kPostOrder>` visits SIBLINGS FORWARD; we visit them backward, so
/// that e466's insertions — always after the conditional it rewrote — cannot invalidate a path not yet
/// processed. The outcome set is unchanged: the only cross-sibling coupling is e465's
/// order-insensitive *"the parent has no more uses"* test.
pub fn run_on_operation<A: Arch, M: Model, W: Workload>(
    program: &mut Program<A, M, W>,
    values: &mut Values,
) {
    if DISABLE_THIS_PASS {
        return;
    }
    for unit in program.units.iter_mut() {
        let mut paths = Vec::new();
        if_paths(&unit.body, &mut Vec::new(), &mut paths);
        let mut to_be_deleted: Vec<Doomed> = Vec::new();
        for if_at in paths.iter().rev() {
            process_if_op(&mut unit.body, if_at, &mut to_be_deleted, values);
        }
        for doomed in to_be_deleted {
            match doomed {
                Doomed::If(result) => erase_defining_op(&mut unit.body, result),
                Doomed::For(ForRef(iv)) => erase_for_op(&mut unit.body, iv),
            }
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::run_on_operation;
    use crate::arch::Dd2;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::dataflow_ir::{GroupId, OpIndex, ProgramName, Units, Values};
    use crate::islands::sentient::dialects::{Op, Val, sentient};
    use crate::islands::sentient::{Program, ProgramUnit, ProgramUnits};
    use crate::model::Model;
    use crate::units::DfirUnit;
    use crate::workload::Workload;

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

    /// A one-unit program running `body`.
    fn one_unit(body: Vec<Op>) -> Program<Dd2, AnyModel, AnyRung> {
        Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            preamble: Vec::new(),
            units: ProgramUnits::of(
                ProgramUnit {
                    on: Units::one(DfirUnit::Lxlu, Val(0)),
                    precision: None,
                    body,
                    arch: core::marker::PhantomData,
                },
                Vec::new(),
            ),
            bound: core::marker::PhantomData,
        }
    }

    /// e575 — e466's own case through the pass entry: the loop guarded by a conditional becomes a
    /// conditional around the loop's body, and BOTH emptied ops are erased by the post-walk drain.
    #[test]
    fn e575_simplifies_every_conditional_then_drains_the_queue() {
        let (a, b, zero, one) = (Val(0), Val(1), Val(2), Val(3));
        let (if_result, iv, inner) = (Val(4), Val(5), Val(6));
        let mut program = one_unit(vec![
            constant(a, 3),
            constant(b, 4),
            constant(zero, 0),
            constant(one, 1),
            Op::Sentient(sentient::Op::If {
                predicate: sentient::CmpPredicate::Slt,
                lhs: a,
                rhs: b,
                yielded: vec![sentient::Yielded {
                    result: if_result,
                    reg: sentient::Reg {
                        locale: sentient::RegType::Unknown,
                        index: None,
                    },
                    element_size: None,
                }],
                dbg_name: None,
                then_body: vec![yield_op(vec![one])],
                else_body: vec![yield_op(vec![zero])],
            }),
            Op::Sentient(sentient::Op::For {
                iv,
                bound: if_result,
                bound_reg: None,
                carried: Vec::new(),
                dbg_name: None,
                body: vec![constant(inner, 7), yield_op(Vec::new())],
            }),
        ]);
        let mut values = Values::default();
        // Val(0)..=Val(6) are taken above.
        for _ in 0..7 {
            let _ = values.mint();
        }

        run_on_operation(&mut program, &mut values);

        let unit = program.units.iter().next().expect("the one unit");
        let body = &unit.body;
        // The four constants, e466's new bound, and the new conditional — the old `if` and the
        // emptied `for` are both gone.
        assert_eq!(body.len(), 6);
        let Op::Sentient(sentient::Op::If { then_body, .. }) = &body[5] else {
            panic!("the new conditional follows the new bound");
        };
        assert_eq!(then_body, &vec![constant(inner, 7), yield_op(Vec::new())]);
        assert!(
            !body
                .iter()
                .any(|op| matches!(op, Op::Sentient(sentient::Op::For { .. })))
        );
    }
}
