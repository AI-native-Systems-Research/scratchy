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

//! `UniformMapCanonicalization.cpp` — 5 of the campaign's 656 units (dependency level(s) [0, 2, 3, 4]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e238_cleanupConstants` | 238 | 0 | 10 | `dcc/src/Transform/Sentient/UniformMapCanonicalization.cpp:105` |
//! | `e484_runOn` | 484 | 2 | 12 | `dcc/src/Transform/Sentient/UniformMapCanonicalization.cpp:83` |
//! | `e544_runOn` | 544 | 3 | 7 | `dcc/src/Transform/Sentient/UniformMapCanonicalization.cpp:75` |
//! | `e545_runOn` | 545 | 3 | 8 | `dcc/src/Transform/Sentient/UniformMapCanonicalization.cpp:96` |
//! | `e586_runOnOperation` | 586 | 4 | 5 | `dcc/src/Transform/Sentient/UniformMapCanonicalization.cpp:58` |

// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET, so everything below is reachable only from this
// file's own tests until `e545_runOn` and `e586_runOnOperation` land. CI runs clippy with
// `-D warnings`, so without this the first ported leaf of the module fails the gate.
// ⭐ REMOVE THIS WITH `e586_runOnOperation`: an unused item here is a real defect again at that point.
#![allow(dead_code)]

use crate::arch::Arch;
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::Program;
use crate::islands::sentient::dialects::{self, Op, Val, sentient};
use crate::model::Model;
use crate::workload::Workload;

/// ONE CONSTANT'S FOLD KEY — `std::make_tuple(op->getDialect(), constValue, *op->result_type_begin())`
/// (`mlir/lib/Transforms/Utils/FoldUtils.cpp`, `OperationFolder::insertKnownConstant`).
///
/// ⛔⛔ `regLocale` IS **NOT** IN THE KEY. Two `sentient.scalar_constant`s of the same value and type
/// unify onto the first one seen even when one is `imm` and the other is not — the locale of the
/// survivor is the one every reader ends up with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ConstId {
    /// `getValue()`, an `si64` attribute.
    value: i64,
    /// `getType()`.
    ty: ScalarTy,
}

/// `m_Constant` FOR THE ONE OP THIS PASS NAMES — `sentient::ConstantOp`.
///
/// ⛔ NO OTHER `ConstantLike` OP IS TOUCHED HERE. `sentient.vector_constant` and the `arith`
/// constants carry the trait too, so the greedy driver would unify them on the same key; the
/// reference's own comment scopes this call to `sentient.scalar_constant`, and an `arith.constant`
/// holding a dense attribute has no key this island can spell (see
/// [`lexical_ordering`](super::lexical_ordering)'s `ConstKey::ArithDense`).
fn scalar_constant(op: &Op) -> Option<(ConstId, Val)> {
    match op {
        Op::Sentient(sentient::Op::ScalarConstant {
            value, result, ty, ..
        }) => Some((
            ConstId {
                value: *value,
                ty: *ty,
            },
            *result,
        )),
        _ => None,
    }
}

/// The driver's pre-order walk, collecting every `sentient.scalar_constant` in the order it is met.
///
/// ⛔ PROVED ABSENT BY THE TYPE: a shared dialect's region holds the rung below's ops, which have no
/// `Sentient` arm — so no scalar constant of this rung can be inside one.
fn collect(ops: &[Op], found: &mut Vec<(ConstId, Val)>) {
    for op in ops {
        if let Some(constant) = scalar_constant(op) {
            found.push(constant);
        }
        match op {
            Op::Sentient(inner) => {
                for region in sentient::regions(inner) {
                    collect(region, found);
                }
            }
            Op::AffineFor(loop_op) => collect(&loop_op.body, found),
            // ⛔ NEITHER `uniformize_regions` NOR `equalize_pattern` IS `IsolatedFromAbove`
            // (`Uniform.td:79-80`, `:187-188`), so a constant a sink put inside one shares the
            // function's fold scope and hoists out of it.
            Op::UniformRegions(regions) => {
                for region in regions.regions() {
                    collect(&region.body, found);
                }
            }
            Op::Dataflow(_)
            | Op::Agen(_)
            | Op::VectorChain(_)
            | Op::Affine(_)
            | Op::Vector(_)
            | Op::Arith(_)
            | Op::Scf(_)
            | Op::Symbol(_)
            | Op::Uniform(_) => {}
        }
    }
}

/// TAKE EACH COLLECTED CONSTANT OUT OF WHEREVER IT SITS — the first half of both `moveBefore` and
/// `erase()`.
fn take_from(ops: &mut Vec<Op>, wanted: &[Val], taken: &mut Vec<(Val, Op)>) {
    let mut at = 0;
    while at < ops.len() {
        if let Some((_, result)) = scalar_constant(&ops[at])
            && wanted.contains(&result)
        {
            taken.push((result, ops.remove(at)));
            continue;
        }
        match &mut ops[at] {
            Op::Sentient(inner) => {
                for region in sentient::regions_mut(inner) {
                    take_from(region, wanted, taken);
                }
            }
            Op::AffineFor(loop_op) => take_from(&mut loop_op.body, wanted, taken),
            // See [`collect`]: a local region is not a fold barrier.
            Op::UniformRegions(regions) => {
                for region in regions.regions_mut() {
                    take_from(&mut region.body, wanted, taken);
                }
            }
            Op::Dataflow(_)
            | Op::Agen(_)
            | Op::VectorChain(_)
            | Op::Affine(_)
            | Op::Vector(_)
            | Op::Arith(_)
            | Op::Scf(_)
            | Op::Symbol(_)
            | Op::Uniform(_) => {}
        }
        at += 1;
    }
}

/// How many operands in the whole program read `of`.
fn uses<A: Arch, M: Model, W: Workload>(of: Val, program: &Program<A, M, W>) -> usize {
    let mut count = dialects::use_count(of, &program.preamble);
    for unit in program.units.iter() {
        count += dialects::use_count(of, &unit.body);
    }
    count
}

/// Replaces: e238_cleanupConstants
///
/// Unifies the program's duplicate `sentient.scalar_constant`s onto their first occurrence, hoists
/// each survivor to the head of the function's entry block and drops the ones nothing reads.
///
/// ⛔⛔ THE PATTERN SET IT BUILDS IS **EMPTY**: `ConstantOp` declares `hasFolder` and NO
/// `hasCanonicalizer` (`SentientOps.td:848-864`), so `getCanonicalizationPatterns` is `mlir::Op`'s
/// do-nothing default and the whole effect is the greedy driver's own constant folding.
/// ⛔ THE SCOPE IS THE **FUNCTION**, NOT THE UNIT: `dataflow.program_unit` is not `IsolatedFromAbove`,
/// so a constant leaves the unit it was written in and dedupes program-wide.
/// ⚠️ RESIDUE: the driver also folds and dead-erases every other op it walks; only the constants this
/// call's own comment names are ported. The pass ships DISABLED (`DisableThisPass`, `:39-42`).
pub(crate) fn cleanup_constants<A: Arch, M: Model, W: Workload>(program: &mut Program<A, M, W>) {
    let mut found: Vec<(ConstId, Val)> = Vec::new();
    collect(&program.preamble, &mut found);
    for unit in program.units.iter() {
        collect(&unit.body, &mut found);
    }

    // `if (folderConstOp) { replaceAllUsesWith(op, folderConstOp); op->erase(); }` — the FIRST
    // occurrence of a key is the survivor and every later one is rewired onto it and erased.
    let mut survivors: Vec<(ConstId, Val)> = Vec::new();
    let mut rewires: Vec<(Val, Val)> = Vec::new();
    for (id, result) in &found {
        match survivors.iter().find(|(kept, _)| kept == id) {
            Some((_, kept)) => rewires.push((*result, *kept)),
            None => survivors.push((*id, *result)),
        }
    }
    for (of, with) in &rewires {
        dialects::replace_all_uses_with(&mut program.preamble, *of, *with);
        for unit in program.units.iter_mut() {
            dialects::replace_all_uses_with(&mut unit.body, *of, *with);
        }
    }
    let duplicates: Vec<Val> = rewires.iter().map(|(of, _)| *of).collect();
    take_everywhere(program, &duplicates);

    // `op->moveBefore(&insertBlock->front())` (`FoldUtils.cpp:160-168`) — ⛔ A SURVIVOR THE ENTRY BLOCK
    // ALREADY HELD DOES NOT MOVE when it heads the block or follows one this walk already uniqued;
    // MLIR takes the pre-existing constants first *"to avoid accidentally reversing the constant
    // order"* (`mlir/../GreedyPatternRewriteDriver.cpp:846-854`). Only the ones imported from a unit
    // or a nested region are pushed to the front, and those land in reverse discovery order.
    let mut uniqued: Vec<Val> = Vec::new();
    for (_, result) in &survivors {
        match at_top_level_of(&program.preamble, *result) {
            Some(0) => {}
            Some(at)
                if scalar_constant(&program.preamble[at - 1])
                    .is_some_and(|(_, before)| uniqued.contains(&before)) => {}
            Some(at) => {
                let op = program.preamble.remove(at);
                program.preamble.insert(0, op);
            }
            None => {
                for (_, op) in take_everywhere(program, &[*result]) {
                    program.preamble.insert(0, op);
                }
            }
        }
        uniqued.push(*result);
    }

    // ⛔ A SURVIVOR NOTHING READS IS TRIVIALLY DEAD — the unique constants DO go on the driver's
    // worklist (`GreedyPatternRewriteDriver.cpp:859`, only the replaced duplicates are skipped), and
    // `processWorklist` erases a trivially dead op before it tries any pattern (`:484-486`).
    let dead: Vec<Val> = survivors
        .iter()
        .map(|(_, result)| *result)
        .filter(|result| uses(*result, program) == 0)
        .collect();
    program
        .preamble
        .retain(|op| !scalar_constant(op).is_some_and(|(_, result)| dead.contains(&result)));
}

/// Where `result`'s `sentient.scalar_constant` sits in the entry block itself, if it is there.
fn at_top_level_of(ops: &[Op], result: Val) -> Option<usize> {
    ops.iter()
        .position(|op| scalar_constant(op).is_some_and(|(_, at)| at == result))
}

/// `op->erase()`, and the first half of `op->moveBefore(..)`, over the whole program.
fn take_everywhere<A: Arch, M: Model, W: Workload>(
    program: &mut Program<A, M, W>,
    wanted: &[Val],
) -> Vec<(Val, Op)> {
    let mut taken: Vec<(Val, Op)> = Vec::new();
    take_from(&mut program.preamble, wanted, &mut taken);
    for unit in program.units.iter_mut() {
        take_from(&mut unit.body, wanted, &mut taken);
    }
    taken
}

// crustify:todo: e484_runOn
//   authority : dcc/src/Transform/Sentient/UniformMapCanonicalization.cpp:83  (12 body lines, level 2)
//   original  : void UniformMapCanonicalizationPass::runOn(Operation *op)
//   calls     : e395_pruneOutOfScopeEntries

// crustify:todo: e544_runOn
//   authority : dcc/src/Transform/Sentient/UniformMapCanonicalization.cpp:75  (7 body lines, level 3)
//   original  : void UniformMapCanonicalizationPass::runOn(dataflow::ProgramUnitOp unit)
//   calls     : e484_runOn, e545_runOn

// crustify:todo: e545_runOn
//   authority : dcc/src/Transform/Sentient/UniformMapCanonicalization.cpp:96  (8 body lines, level 3)
//   original  : void UniformMapCanonicalizationPass::runOn(ModuleOp module_op)
//   calls     : e238_cleanupConstants, e484_runOn, e544_runOn

// crustify:todo: e586_runOnOperation
//   authority : dcc/src/Transform/Sentient/UniformMapCanonicalization.cpp:58  (5 body lines, level 4)
//   original  : void runOnOperation()
//   calls     : e484_runOn, e544_runOn, e545_runOn

#[cfg(test)]
mod unit_tests {
    use super::cleanup_constants;
    use crate::arch::Dd2;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::dataflow_ir::{GroupId, OpIndex, ProgramName, Units};
    use crate::islands::sentient::dialects::{Op, Val, sentient};
    use crate::islands::sentient::{Program, ProgramUnit, ProgramUnits};
    use crate::model::Model;
    use crate::units::DfirUnit;
    use crate::workload::Workload;

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

    /// `%result = sentient.scalar_constant value` in `locale`.
    fn constant(value: i64, result: Val, locale: sentient::RegType) -> Op {
        Op::Sentient(sentient::Op::ScalarConstant {
            value,
            result,
            reg_locale: locale,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    /// `%result = sentient.scalar_add %lhs, %lhs` — a reader, so a constant is not dead.
    fn adds(lhs: Val, result: Val) -> Op {
        Op::Sentient(sentient::Op::ScalarAdd {
            lhs,
            rhs: lhs,
            result,
            reg: None,
            element_size: None,
            ty: ScalarTy::Index,
        })
    }

    /// A one-unit program running `body`.
    fn program_of(body: Vec<Op>) -> Program<Dd2, AnyModel, AnyRung> {
        Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            preamble: Vec::new(),
            units: ProgramUnits::of(
                ProgramUnit {
                    iter_arg: None,
                    on: Units::one(DfirUnit::Lxsu, Val(0)),
                    precision: None,
                    body,
                    arch: core::marker::PhantomData,
                },
                Vec::new(),
            ),
            bound: core::marker::PhantomData,
        }
    }

    /// e238 — two same-valued constants unify onto the first even across a `sentient.for` and across
    /// a differing `regLocale`, the survivors are hoisted out of the unit into the entry block in
    /// reverse discovery order, and the one nothing reads is dropped.
    #[test]
    fn e238_unifies_and_hoists_the_scalar_constants() {
        let mut program = program_of(vec![
            constant(7, Val(1), sentient::RegType::Imm),
            constant(9, Val(2), sentient::RegType::Imm),
            adds(Val(1), Val(3)),
            Op::Sentient(sentient::Op::For {
                iv_reg: sentient::Reg::UNALLOCATED,
                iv: Val(4),
                bound: Val(2),
                carried: Vec::new(),
                dbg_name: None,
                body: vec![
                    constant(7, Val(5), sentient::RegType::Lrf),
                    adds(Val(5), Val(6)),
                ],
            }),
        ]);
        cleanup_constants(&mut program);

        // `9` bound nothing but the loop's trip count, which reads it — `7` survives once, in `imm`.
        assert_eq!(
            program.preamble,
            vec![
                constant(9, Val(2), sentient::RegType::Imm),
                constant(7, Val(1), sentient::RegType::Imm),
            ]
        );
        assert_eq!(
            program.units.iter().next().expect("the head unit").body,
            vec![
                adds(Val(1), Val(3)),
                Op::Sentient(sentient::Op::For {
                    iv_reg: sentient::Reg::UNALLOCATED,
                    iv: Val(4),
                    bound: Val(2),
                    carried: Vec::new(),
                    dbg_name: None,
                    // ⛔ THE DUPLICATE'S READER MOVED ONTO THE SURVIVOR, which is now in the entry
                    // block — the `lrf` locale it was written with is gone with it.
                    body: vec![adds(Val(1), Val(6))],
                }),
            ]
        );
    }

    /// e238 — ⛔ THE ENTRY BLOCK'S OWN CONSTANTS ARE NOT REORDERED: a leading run of them is left
    /// exactly where it stands, and only the one imported from the unit is pushed in front of it.
    #[test]
    fn e238_leaves_the_entry_blocks_leading_constants_in_place() {
        let mut program = program_of(vec![
            constant(5, Val(3), sentient::RegType::Lrf),
            adds(Val(3), Val(12)),
        ]);
        program.preamble = vec![
            constant(7, Val(1), sentient::RegType::Imm),
            constant(9, Val(2), sentient::RegType::Imm),
            adds(Val(1), Val(10)),
            adds(Val(2), Val(11)),
        ];
        cleanup_constants(&mut program);

        assert_eq!(
            program.preamble,
            vec![
                constant(5, Val(3), sentient::RegType::Lrf),
                constant(7, Val(1), sentient::RegType::Imm),
                constant(9, Val(2), sentient::RegType::Imm),
                adds(Val(1), Val(10)),
                adds(Val(2), Val(11)),
            ]
        );
        assert_eq!(
            program.units.iter().next().expect("the head unit").body,
            vec![adds(Val(3), Val(12))]
        );
    }
}
