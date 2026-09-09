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

//! `SimplifyUniformRegions.cpp` — 1 of the campaign's 656 units (dependency level(s) [0]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e208_runOnOperation` | 208 | 0 | 51 | `dcc/src/Transform/Sentient/SimplifyUniformRegions.cpp:55` |


// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET, so everything below is reachable only from this
// file's own tests until a pipeline calls it. CI runs clippy with `-D warnings`, so without this the
// module's first ported unit fails the gate.
// ⭐ REMOVE THIS WHEN THE PASS IS WIRED: at that point an unused item here is a real defect again.
#![allow(dead_code)]

use crate::arch::Arch;
use crate::islands::sentient::Program;
use crate::islands::sentient::dialects::{self, Op, UniformRegions, uniform};
use crate::model::Model;
use crate::workload::Workload;

/// `-dcc-simplify-uniform-regions-disable`, `cl::init(false)` (`SimplifyUniformRegions.cpp:36-39`).
const DISABLE_THIS_PASS: bool = false;

/// Replaces: e208_runOnOperation
///
/// Four walks: flatten every uniform region in the module; per program unit merge its local ops and —
/// unless disabled — drop the empty and the unnecessary ones and simplify the rest; then canonicalise
/// every query map.
///
/// ⛔ ALL SIX CALLEES ARE `dcc/src/Dialect/Uniform/Utils.cpp` AND OUT OF CAMPAIGN SCOPE, so the
/// traversal, the ordering and the `DisableThisPass` gate are ported and each call site is a `todo!`
/// naming its authority line. ⭐ TWO OF THEM ARE ALREADY PORTED ONE RUNG DOWN — bridge 2's private
/// `flatten_uniform_region` and `simplify_query_map_with_same_target` — but both take a
/// `dataflow_ir::dialects::Op` and the first mints values from a `Values` this rung's [`Program`] does
/// not carry, so calling them from here is its own change and not this unit's.
///
/// ⛔ TRAP: EVERY `WalkResult::skip()` IN THE REFERENCE IS INERT. `Operation::walk` defaults to
/// `WalkOrder::PostOrder` (`mlir/IR/Visitors.h:272`), which calls back only AFTER the nested regions
/// have been walked (`:234-246`), so nothing is pruned and a nested uniform op is visited FIRST.
///
/// ⭐ `dcc::getUnitType(prog_unit_op)` NEEDS NO WALK: [`crate::islands::dataflow_ir::Units`] binds the
/// kind to the unit list, so `unit.on.kind()` is the answer.
pub(crate) fn run_on_program<A: Arch, M: Model, W: Workload>(program: &mut Program<A, M, W>) {
    // THE FIRST WALK (`:57-61`) — the whole module, so the preamble as well as every unit body.
    let mut flatten = |op: &Op| {
        if is_uniformize_regions(op) {
            todo!(
                "dcc::uniform::utils::flattenUniformRegion (Dialect/Uniform/Utils.cpp:563) — out of \
                 campaign scope; bridge 2 ports it at the DataflowIR rung as the private \
                 `flatten_uniform_region`, which mints values from a `Values` this rung does not carry"
            )
        }
    };
    walk_post_order(&program.preamble, &mut flatten);
    for unit in program.units.iter() {
        walk_post_order(&unit.body, &mut flatten);
    }

    // THE SECOND WALK (`:62-101`) — per `dataflow.program_unit`.
    for unit in program.units.iter() {
        // `mergeAllLocalOps(prog_unit_op)` (`:64`) is ITSELF a walk (`Utils.cpp:1239-1261`) whose body
        // fires only on a local op, so a unit holding none is genuinely untouched and the seam is
        // where the sister-merging loop would begin.
        walk_post_order(&unit.body, &mut |op| {
            if is_local_op(op) {
                todo!(
                    "dcc::uniform::utils::{{mergeLocalOps,getNextSisterOperation}} \
                     (Dialect/Uniform/Utils.cpp:1187,1164) — out of campaign scope: \
                     `mergeAllLocalOps`' merge of this unit's consecutive local ops (`:1239-1261`)"
                )
            }
        });

        if !DISABLE_THIS_PASS {
            walk_post_order(&unit.body, &mut |op| {
                if is_uniformize_regions(op) {
                    todo!(
                        "dcc::uniform::utils::removeEmptyLocalRegions \
                         (Dialect/Uniform/Utils.cpp:1023) — out of campaign scope"
                    )
                }
            });
            walk_post_order(&unit.body, &mut |op| {
                if is_uniformize_regions(op) {
                    todo!(
                        "dcc::uniform::utils::removeNotNecessaryLocalRegions \
                         (Dialect/Uniform/Utils.cpp:1007) — out of campaign scope: promoting a local \
                         region's operations to global where it can"
                    )
                }
            });

            // ⭐ `PropagationAnalysis` IS NOT NEEDED HERE, and the reference is why: its lookup is
            // COMMENTED OUT (`:84-89`) and `simplifyUniformRegions` is passed `nullptr` (`:94`).
            let comp = unit.on.kind();
            walk_post_order(&unit.body, &mut |op| {
                if is_local_op(op) {
                    todo!(
                        "dcc::uniform::utils::simplifyUniformRegions \
                         (Dialect/Uniform/Utils.cpp:994) — out of campaign scope: the whole \
                         UniformRegionSimplicationHelper over this {comp:?} unit's local op, with no \
                         propagation analysis (SimplifyUniformRegions.cpp:84-89,94)"
                    )
                }
            });
        }
    }

    // THE THIRD WALK (`:102-104`) — every `uniform.query_map` in the module.
    let mut simplify_query_maps = |op: &Op| {
        if matches!(op, Op::Uniform(uniform::Op::QueryMap { .. })) {
            todo!(
                "dcc::uniform::utils::simplifyQueryMapWithSameTarget \
                 (Dialect/Uniform/Utils.cpp:1263) — out of campaign scope; bridge 2 ports it at the \
                 DataflowIR rung as the private `simplify_query_map_with_same_target`"
            )
        }
    };
    walk_post_order(&program.preamble, &mut simplify_query_maps);
    for unit in program.units.iter() {
        walk_post_order(&unit.body, &mut simplify_query_maps);
    }
}

/// `llvm::dyn_cast<uniform::UniformizeRegionsOp>(op)` (`:58`), which three of the four walks test for.
///
/// ⛔ BOTH ISLAND SPELLINGS OF THE ONE OP. `dyn_cast` answers on the operation, never on what its
/// regions hold, so [`Op::UniformRegions`] — the same `uniform.uniformize_regions` with THIS rung's
/// ops inside, which is exactly what `SinkScalarCopy` mints — matches as well. Testing only
/// [`Op::Uniform`] made all four walks a silent no-op on that shape.
fn is_uniformize_regions(op: &Op) -> bool {
    matches!(
        op,
        Op::Uniform(uniform::Op::UniformizeRegions { .. })
            | Op::UniformRegions(UniformRegions::UniformizeRegions { .. })
    )
}

/// `isa<uniform::UniformizeRegionsOp, uniform::EqualizePatternOp>(op)` — the pair the merge
/// (`Utils.cpp:1242-1243`) and `simplifyUniformRegions` (`:92`) both accept, in both island spellings
/// (see [`is_uniformize_regions`]).
fn is_local_op(op: &Op) -> bool {
    matches!(
        op,
        Op::Uniform(uniform::Op::UniformizeRegions { .. } | uniform::Op::EqualizePattern { .. })
            | Op::UniformRegions(
                UniformRegions::UniformizeRegions { .. } | UniformRegions::EqualizePattern { .. }
            )
    )
}

/// `Operation::walk` IN ITS DEFAULT `WalkOrder::PostOrder` — nested first, then the op itself.
///
/// ⛔ AN [`Op::Uniform`] REGION HOLDS RUNG-BELOW OPS AND CANNOT BE ENTERED FROM HERE — see
/// [`dialects::regions_ref`], which is the same reason the two ported utils live at the DataflowIR
/// rung. An [`Op::UniformRegions`] region holds THIS rung's ops and IS entered, so a local region's
/// contents are walked before the op carrying them.
fn walk_post_order(scope: &[Op], visit: &mut impl FnMut(&Op)) {
    for op in scope {
        for region in dialects::regions_ref(op) {
            walk_post_order(region, visit);
        }
        visit(op);
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Dd2;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::{GroupId, OpIndex, ProgramName, Units};
    use crate::islands::sentient::dialects::{LocalRegion, Val, sentient};
    use crate::islands::sentient::{ProgramUnit, ProgramUnits};
    use crate::units::DfirUnit;

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

    /// A one-unit program running `body`.
    fn program(body: Vec<Op>) -> Program<Dd2, AnyModel, AnyRung> {
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

    /// e208 — the module walk comes first, so a program holding a local region stops at the flatten.
    #[test]
    #[should_panic(expected = "flattenUniformRegion")]
    fn e208_flattens_before_it_merges() {
        let mut held = program(vec![Op::Uniform(uniform::Op::UniformizeRegions {
            regions: Vec::new(),
            results: Vec::new(),
        })]);
        run_on_program(&mut held);
    }

    /// e208 — `isa<>` answers on the OP, so the sentient-rung spelling of the same
    /// `uniform.uniformize_regions` — the shape `SinkScalarCopy` mints — reaches the flatten too.
    #[test]
    #[should_panic(expected = "flattenUniformRegion")]
    fn e208_flattens_a_local_region_holding_this_rungs_ops() {
        let mut sunk = program(vec![Op::UniformRegions(
            UniformRegions::UniformizeRegions {
                regions: vec![LocalRegion {
                    arg: Val(1),
                    units: vec![Val(0)],
                    body: vec![Op::Sentient(sentient::Op::Nop { dbg_name: None })],
                }],
                results: Vec::new(),
            },
        )]);
        run_on_program(&mut sunk);
    }

    /// e208 — a program with no uniform op anywhere runs all four walks to completion and changes
    /// nothing, because every one of them fires only on a `uniform` op.
    #[test]
    fn e208_leaves_a_program_with_no_uniform_op_alone() {
        let nop = Op::Sentient(sentient::Op::Nop { dbg_name: None });
        let mut plain = program(vec![nop.clone()]);
        run_on_program(&mut plain);
        assert_eq!(
            plain.units.iter().next().expect("the head unit").body,
            vec![nop]
        );
    }
}
