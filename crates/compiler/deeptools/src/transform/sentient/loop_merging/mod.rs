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

//! `LoopMerging.cpp` — 7 of the campaign's 656 units (dependency level(s) [0, 1, 3, 4, 5, 6]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e075_getNextEligibleOp` | 075 | 0 | 11 | `dcc/src/Transform/Sentient/LoopMerging.cpp:63` |
//! | `e076_getNewDbgName` | 076 | 0 | 6 | `dcc/src/Transform/Sentient/LoopMerging.cpp:75` |
//! | `e316_mergeLoops` | 316 | 1 | 100 | `dcc/src/Transform/Sentient/LoopMerging.cpp:179` |
//! | `e508_loopsAreMergeable` | 508 | 3 | 94 | `dcc/src/Transform/Sentient/LoopMerging.cpp:83` |
//! | `e564_runLoopMerging` | 564 | 4 | 39 | `dcc/src/Transform/Sentient/LoopMerging.cpp:281` |
//! | `e601_runOn` | 601 | 5 | 13 | `dcc/src/Transform/Sentient/LoopMerging.cpp:321` |
//! | `e627_runOnOperation` | 627 | 6 | 5 | `dcc/src/Transform/Sentient/LoopMerging.cpp:335` |

use crate::islands::sentient::dialects::{Op, dataflow, sentient, symbol, uniform};

/// WHERE AN OP SITS IN ITS BLOCK — the `Operation *` identity `loopsAreMergeable` compares against
/// the second loop (`:85`).
///
/// ⛔ A POSITION AND NOT A BORROW, because `getNextEligibleOp`'s answer is used for IDENTITY
/// (`getNextEligibleOp(loop_a) != loop_b`), and this island's ops are a tree with no addresses to
/// compare.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InBlock(pub usize);

/// THE FIVE OPS MERGING STEPS OVER — `dataflow.get_unit`, `sentient.scalar_constant`,
/// `symbol.create_symbol`, `uniform.def_immutable_mapping` and `uniform.query_map` (`:66-69`).
///
/// ⛔ IT IS `uniform.query_map`, NOT `symbol.query_map`: the symbol dialect declares one of the same
/// name and the reference names the uniform one, so a symbol query between two loops keeps them
/// apart.
fn skipped_between_loops(op: &Op) -> bool {
    matches!(
        op,
        Op::Dataflow(dataflow::Op::GetUnit { .. })
            | Op::Sentient(sentient::Op::ScalarConstant { .. })
            | Op::Symbol(symbol::Op::CreateSymbol { .. })
            | Op::Uniform(uniform::Op::DefImmutableMapping { .. } | uniform::Op::QueryMap { .. })
    )
}

/// Replaces: e075_getNextEligibleOp
///
/// The next op after `op` in its own block, walking past the bookkeeping ops of
/// [`skipped_between_loops`].
///
/// ⛔ TRAP: `None` IS THE REFERENCE'S NULL AND ITS CALLER TREATS IT AS "NOT ADJACENT" — the last op
/// of a block is a merge candidate with nothing, so returning the block's end here would make two
/// loops in different blocks look adjacent.
#[must_use]
pub fn get_next_eligible_op(block: &[Op], op: InBlock) -> Option<InBlock> {
    block
        .iter()
        .enumerate()
        .skip(op.0 + 1)
        .find(|(_, next)| !skipped_between_loops(next))
        .map(|(at, _)| InBlock(at))
}

/// A LOOP'S `dbgName`, PRESENT — `for_op.getDbgName().value()` (`:77-78`) as a type.
///
/// ⛔ `.value()` ON AN ABSENT NAME IS `std::bad_optional_access`, so the reference merges two
/// unnamed loops by throwing. Here a nameless loop cannot reach [`get_new_dbg_name`] at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DbgName(String);

impl DbgName {
    /// The name a `sentient.for` carries, or nothing when it is not a loop or carries none.
    #[must_use]
    pub fn of(op: &Op) -> Option<DbgName> {
        let Op::Sentient(sentient::Op::For { dbg_name, .. }) = op else {
            return None;
        };
        dbg_name.clone().map(DbgName)
    }

    /// The text it carries.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.0
    }
}

/// Replaces: e076_getNewDbgName
///
/// `LM(<first>, <second>)` — the merged loop's `dbgName`, built from both sources' own.
#[must_use]
pub fn get_new_dbg_name(for_op1: &DbgName, for_op2: &DbgName) -> DbgName {
    DbgName(format!("LM({}, {})", for_op1.text(), for_op2.text()))
}

// crustify:todo: e316_mergeLoops
//   authority : dcc/src/Transform/Sentient/LoopMerging.cpp:179  (100 body lines, level 1)
//   original  : void LoopMergingPass::mergeLoops( std::pair<dcc::LoopNode *, dcc::LoopNode *> loops)
//   calls     : e075_getNextEligibleOp, e252_size

// crustify:todo: e508_loopsAreMergeable
//   authority : dcc/src/Transform/Sentient/LoopMerging.cpp:83  (94 body lines, level 3)
//   original  : bool LoopMergingPass::loopsAreMergeable(sentient::ForOp loop_a, sentient::ForOp loop_b)
//   calls     : e075_getNextEligibleOp, e252_size, e422_insert

// crustify:todo: e564_runLoopMerging
//   authority : dcc/src/Transform/Sentient/LoopMerging.cpp:281  (39 body lines, level 4)
//   original  : void LoopMergingPass::runLoopMerging(Operation *op)
//   calls     : e316_mergeLoops, e508_loopsAreMergeable

// crustify:todo: e601_runOn
//   authority : dcc/src/Transform/Sentient/LoopMerging.cpp:321  (13 body lines, level 5)
//   original  : void LoopMergingPass::runOn(ModuleOp module_op)
//   calls     : e564_runLoopMerging

// crustify:todo: e627_runOnOperation
//   authority : dcc/src/Transform/Sentient/LoopMerging.cpp:335  (5 body lines, level 6)
//   original  : void LoopMergingPass::runOnOperation()
//   calls     : e601_runOn

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::Val;
    use crate::units::{DfirUnit, Residency};

    /// `sentient.for` carrying `dbg_name`.
    fn sentient_for(dbg_name: Option<&str>) -> Op {
        Op::Sentient(sentient::Op::For {
            iv: Val(0),
            bound: Val(1),
            carried: Vec::new(),
            dbg_name: dbg_name.map(str::to_owned),
            body: Vec::new(),
        })
    }

    /// The four non-`sentient` bookkeeping ops the walk steps over, plus the constant.
    fn skipped_ops() -> Vec<Op> {
        vec![
            Op::Dataflow(dataflow::Op::GetUnit {
                result: Val(2),
                residency: Residency::Global,
                unit: DfirUnit::L3lu,
                num_folds: None,
            }),
            Op::Sentient(sentient::Op::ScalarConstant {
                value: 0,
                result: Val(3),
                reg_locale: sentient::RegType::Imm,
                ty: ScalarTy::Index,
                is_symbol: false,
            }),
            Op::Symbol(symbol::Op::CreateSymbol {
                result: Val(4),
                symbol_id: 0,
                max_value: None,
            }),
            Op::Uniform(uniform::Op::DefImmutableMapping {
                result: Val(5),
                pairs: Vec::new(),
            }),
            Op::Uniform(uniform::Op::QueryMap {
                result: Val(6),
                map: Val(5),
                key: Val(3),
            }),
        ]
    }

    /// Two loops separated by all five skipped ops are adjacent; one separated by anything else is
    /// not, and the last op of a block is adjacent to nothing.
    #[test]
    fn e075_walks_past_the_five_bookkeeping_ops_and_stops_at_the_block_end() {
        let mut block = vec![sentient_for(None)];
        block.extend(skipped_ops());
        block.push(sentient_for(None));
        assert_eq!(get_next_eligible_op(&block, InBlock(0)), Some(InBlock(6)));
        assert_eq!(get_next_eligible_op(&block, InBlock(6)), None);

        let blocked = vec![
            sentient_for(None),
            Op::Sentient(sentient::Op::Nop { dbg_name: None }),
            sentient_for(None),
        ];
        assert_eq!(get_next_eligible_op(&blocked, InBlock(0)), Some(InBlock(1)));
    }

    /// The merged name, and the absent name the reference's `.value()` throws on.
    #[test]
    fn e076_wraps_both_source_names_in_lm() {
        let first = sentient_for(Some("L0"));
        let second = sentient_for(Some("LM(L1, L2)"));
        let first = DbgName::of(&first).expect("a named loop");
        let second = DbgName::of(&second).expect("a named loop");
        assert_eq!(
            get_new_dbg_name(&first, &second).text(),
            "LM(L0, LM(L1, L2))"
        );
        assert!(DbgName::of(&sentient_for(None)).is_none());
        assert!(DbgName::of(&Op::Sentient(sentient::Op::Nop { dbg_name: None })).is_none());
    }
}
