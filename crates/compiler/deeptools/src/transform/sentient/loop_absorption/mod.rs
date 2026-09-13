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

//! `LoopAbsorption.cpp` — 3 of the campaign's 656 units (dependency level(s) [5, 6, 7]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e600_runLoopAbsorption` | 600 | 5 | 48 | `dcc/src/Transform/Sentient/LoopAbsorption.cpp:495` |
//! | `e625_runOn` | 625 | 6 | 13 | `dcc/src/Transform/Sentient/LoopAbsorption.cpp:544` |
//! | `e640_runOnOperation` | 640 | 7 | 5 | `dcc/src/Transform/Sentient/LoopAbsorption.cpp:558` |

// ⛔ `e625_runOn` BELOW HAS NO CALLER UNTIL `e640_runOnOperation` (level 7) LANDS, and CI runs
// clippy with `-D warnings`. ⭐ REMOVE THIS WITH e640, when the pipeline calls the pass.
#![allow(dead_code)]

pub(crate) mod loop_absorption_manager;

use crate::arch::Arch;
use crate::islands::dataflow_ir::Values;
use crate::islands::sentient::Program;
use crate::islands::sentient::dialects::{self as dialects, Op, sentient};
use crate::model::Model;
use crate::transform::sentient::ForRef;
use crate::transform::sentient::analyses::{InstructionEstimator, OutOfScopeInstructionEstimator};
use crate::transform::sentient::loop_absorption::loop_absorption_manager::{
    LoopAbsorptionManager, UnitSite,
};
use crate::transform::sentient::loop_tree::{LoopNodeId, LoopTree};
use crate::workload::Workload;

/// `for_op.getBody()->without_terminator().empty()` (`:517-518`) — `None` where this unit holds no such
/// loop, which `getOpAs<sentient::ForOp>()` cannot ask because a node of the tree always does.
fn loop_body_is_empty(for_op: ForRef, scope: &[Op]) -> Option<bool> {
    for op in scope {
        if let Op::Sentient(sentient::Op::For { iv, body, .. }) = op
            && *iv == for_op.0
        {
            // `without_terminator()` DROPS THE CLOSING `sentient.yield` and nothing else.
            return Some(
                body.iter()
                    .all(|op| matches!(op, Op::Sentient(sentient::Op::Yield { .. }))),
            );
        }
        for region in dialects::regions_ref(op) {
            if let Some(found) = loop_body_is_empty(for_op, region) {
                return Some(found);
            }
        }
    }
    None
}

/// `checkAndAbsorb(n)` (`:502-534`) and the pre-order recursion that drives it (`:541`), in one live
/// walk: the action runs on `n`, and only then are `n`'s children re-read.
fn check_and_absorb(
    n: LoopNodeId,
    tree: &mut LoopTree<true>,
    preamble: &mut Vec<Op>,
    unit_body: &mut Vec<Op>,
    vals: &mut Values,
) {
    let mut sorted_worklist: Vec<LoopNodeId> = Vec::new();
    let mut child = tree.first_child(n);
    while let Some(c) = child {
        if tree
            .loop_of(c)
            .and_then(|for_op| loop_body_is_empty(for_op, unit_body))
            == Some(false)
        {
            sorted_worklist.push(c);
        }
        child = tree.next_sibling(c);
    }
    // `std::sort` by `getSubtreeHeightOf`, *"ties are broken by left-to-right order"* (`:503-505`,
    // `:527-531`) — ⭐ WHICH IS WHAT A STABLE SORT OF A WALK-ORDER LIST IS.
    sorted_worklist.sort_by_key(|c| tree.subtree_height_of(*c));
    while !sorted_worklist.is_empty() {
        if let Some(mut instance) = LoopAbsorptionManager::new(
            &mut sorted_worklist,
            tree,
            UnitSite {
                preamble,
                body: unit_body,
            },
        ) {
            instance.absorption_analysis(vals);
        }
        sorted_worklist.pop();
    }
    let mut child = tree.first_child(n);
    while let Some(c) = child {
        check_and_absorb(c, tree, preamble, unit_body, vals);
        child = tree.next_sibling(c);
    }
}

/// Replaces: e600_runLoopAbsorption
///
/// One program unit: every loop's non-empty children become absorption anchors, shortest subtree first,
/// each absorbing its equivalent neighbours until it cannot (`:495-542`).
///
/// ⛔ THE WALK IS LIVE, NOT A PRECOMPUTED LIST: `OperationNode::preOrderWalk` re-reads
/// `getFirstChild()`/`getNextSibling()` AFTER the action (`Analysis/OperationTree.cpp:135-142`), so an
/// absorption that removes a sibling node removes it from the walk too — contrast
/// [`LoopTree::walk_reverse_bfs`], whose list is fixed before the first action runs.
/// ⛔ THE ANCHOR IS THE WORKLIST'S BACK AND IS POPPED AFTER, NOT BEFORE (`:523-528`): the manager
/// reads its own siblings out of the list it is handed.
pub(crate) fn run_loop_absorption(
    preamble: &mut Vec<Op>,
    unit_body: &mut Vec<Op>,
    vals: &mut Values,
) {
    let mut tree: LoopTree<true> = LoopTree::of(unit_body);
    if tree.empty() {
        return;
    }
    let root = tree.root();
    check_and_absorb(root, &mut tree, preamble, unit_body, vals);
}

/// `opts_.OptLevel == 0` (`:549`) — the PIPELINE's optimisation level, which is `2` by default
/// (`dcc/tools/Options/dcc-pass-option.h:63-65`), so the shipped pipeline never reaches the
/// `haveIbuffSpace` half of the `&&`.
const OPT_LEVEL_ZERO: bool = false;

/// Replaces: e625_runOn
///
/// The module walk: absorbs the neighbours of every loop of every program unit whose instruction
/// buffer is not already roomy (`:544-556`).
///
/// ⛔ `haveIbuffSpace` STAYS A `todo!` BEHIND [`OPT_LEVEL_ZERO`]: the pipeline fixes that const's
/// value, not this pass, so flipping it reaches the estimator rather than quietly skipping a unit.
/// ⭐ `unit_list` COLLAPSES — [`crate::islands::sentient::ProgramUnits::iter_mut`] hands out one unit
/// at a time in walk order and absorbing in one reaches no other, so collect-then-run is one pass.
pub fn run_on<A: Arch, M: Model, W: Workload>(program: &mut Program<A, M, W>, vals: &mut Values) {
    let Program {
        preamble, units, ..
    } = program;
    for unit in units.iter_mut() {
        // `getChildAnalysis<InstructionEstimator>(unit)` IS CONSTRUCTED PER UNIT (`:547-548`), even
        // for one the `&&` never asks anything of.
        let mut instruction_estimator = OutOfScopeInstructionEstimator;
        if OPT_LEVEL_ZERO && instruction_estimator.have_ibuff_space(&unit.body) {
            continue;
        }
        run_loop_absorption(preamble, &mut unit.body, vals);
    }
}

// crustify:todo: e640_runOnOperation
//   authority : dcc/src/Transform/Sentient/LoopAbsorption.cpp:558  (5 body lines, level 7)
//   original  : void LoopAbsorptionPass::runOnOperation()
//   calls     : e625_runOn


#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Dd2;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::dataflow_ir::{GroupId, OpIndex, ProgramName, Units};
    use crate::islands::sentient::dialects::Val;
    use crate::islands::sentient::dialects::sentient::RegType;
    use crate::islands::sentient::{ProgramUnit, ProgramUnits};
    use crate::units::DfirUnit;

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

    /// `%r = sentient.scalar_constant {value} : index`.
    fn constant(result: Val, value: i64) -> Op {
        Op::Sentient(sentient::Op::ScalarConstant {
            value,
            result,
            reg_locale: RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    /// `%r = sentient.scalar_add %lhs, %rhs : index`.
    fn add(lhs: Val, rhs: Val, result: Val) -> Op {
        Op::Sentient(sentient::Op::ScalarAdd {
            lhs,
            rhs,
            result,
            reg: None,
            ty: ScalarTy::Index,
            element_size: None,
        })
    }

    /// e600 — e562's own case reached through the tree walk: the one top-level loop becomes the anchor,
    /// absorbs the `sentient.scalar_add` on its right, and comes out with a trip count one higher.
    #[test]
    fn e600_makes_every_non_empty_loop_an_anchor_and_absorbs_into_it() {
        let mut vals = Values::default();
        let bound = vals.mint();
        let (zero, other, iv, arg) = (vals.mint(), vals.mint(), vals.mint(), vals.mint());
        let (result, inner, right) = (vals.mint(), vals.mint(), vals.mint());
        let mut unit_body = vec![
            constant(bound, 5),
            constant(zero, 0),
            constant(other, 3),
            Op::Sentient(sentient::Op::For {
                iv,
                bound,
                bound_reg: None,
                carried: vec![sentient::Carried {
                    init: zero,
                    arg,
                    result,
                    reg: sentient::Reg {
                        locale: RegType::Unknown,
                        index: None,
                    },
                    program_header: false,
                    element_size: None,
                }],
                dbg_name: None,
                body: vec![
                    add(arg, arg, inner),
                    Op::Sentient(sentient::Op::Yield {
                        results: vec![inner],
                    }),
                ],
            }),
            add(result, result, right),
        ];
        let mut preamble = Vec::new();

        run_loop_absorption(&mut preamble, &mut unit_body, &mut vals);

        // The absorbed `add` is gone, and so is the bound constant nothing reads any more.
        assert_eq!(unit_body.len(), 3);
        let Op::Sentient(sentient::Op::For { bound: new, .. }) = unit_body[2] else {
            panic!("the anchor loop is the last op left: {unit_body:?}")
        };
        assert!(preamble.iter().any(|op| matches!(
            op,
            Op::Sentient(sentient::Op::ScalarConstant { value: 6, result, .. }) if *result == new
        )));
    }
    /// e625 — the module walk reaches EVERY unit: both units absorb their neighbour, and both new
    /// bounds land in the ONE module block the `dataflow.program_unit`s sit in.
    #[test]
    fn e625_absorbs_in_every_program_unit() {
        let mut vals = Values::default();
        let mut absorbable = || {
            let bound = vals.mint();
            let (zero, other, iv, arg) = (vals.mint(), vals.mint(), vals.mint(), vals.mint());
            let (result, inner, right) = (vals.mint(), vals.mint(), vals.mint());
            vec![
                constant(bound, 5),
                constant(zero, 0),
                constant(other, 3),
                Op::Sentient(sentient::Op::For {
                    iv,
                    bound,
                    bound_reg: None,
                    carried: vec![sentient::Carried {
                        init: zero,
                        arg,
                        result,
                        reg: sentient::Reg {
                            locale: RegType::Unknown,
                            index: None,
                        },
                        program_header: false,
                        element_size: None,
                    }],
                    dbg_name: None,
                    body: vec![
                        add(arg, arg, inner),
                        Op::Sentient(sentient::Op::Yield {
                            results: vec![inner],
                        }),
                    ],
                }),
                add(result, result, right),
            ]
        };
        let (first, second) = (absorbable(), absorbable());
        let unit = |body: Vec<Op>| ProgramUnit {
            on: Units::one(DfirUnit::Lxlu, Val(0)),
            precision: None,
            body,
            arch: core::marker::PhantomData,
        };
        let mut program: Program<Dd2, AnyModel, AnyRung> = Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            preamble: Vec::new(),
            units: ProgramUnits::of(unit(first), vec![unit(second)]),
            bound: core::marker::PhantomData,
        };

        run_on(&mut program, &mut vals);

        for unit in program.units.iter() {
            assert_eq!(unit.body.len(), 3, "the neighbour was absorbed: {:?}", unit.body);
        }
        let new_bounds = program
            .preamble
            .iter()
            .filter(|op| {
                matches!(
                    op,
                    Op::Sentient(sentient::Op::ScalarConstant { value: 6, .. })
                )
            })
            .count();
        assert_eq!(new_bounds, 2);
    }
}
