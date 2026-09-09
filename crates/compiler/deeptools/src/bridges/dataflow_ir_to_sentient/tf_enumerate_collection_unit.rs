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

//! `EnumerateCollectionUnit.cpp` — 2 of bridge 2's 384 functions (dependency level(s) [2, 3]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e245_enumerateCollectionUnit` | 245/384 | 57 | `dcc/src/Transform/Dataflow/EnumerateCollectionUnit.cpp:34` |
//! | `e286_runOnOperation` | 286/384 | 32 | `dcc/src/Transform/Dataflow/EnumerateCollectionUnit.cpp:94` |

use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::dialects::{
    Op as DfirOp, Val, arith, dataflow, defining_op, regions, regions_mut, replace_uses_of_with,
    results, results_mut,
};

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 245/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// `it.clone(operandMap)` OVER A WHOLE BLOCK — the mapping the reference accumulates across the loop
/// (`EnumerateCollectionUnit.cpp:76-80`), which [`crate::islands::dataflow_ir::dialects::clone_with_fresh_results`]
/// deliberately does not do: one clone there, no remap, no regions.
///
/// ⛔ ORDER IS THE WHOLE OF IT. An op's operands are remapped against the values cloned BEFORE it,
/// then its own fresh results join the map — so a body of N ops that read each other comes out
/// self-consistent and shares nothing with the original.
///
/// ⚠️ A REGION'S BLOCK ARGUMENTS ARE NOT REMINTED, only the results inside it. The collection bodies
/// this runs on carry `dataflow.program_unit`-style regions that bind none.
fn clone_block(body: &[DfirOp], vals: &mut Values, remap: &mut Vec<(Val, Val)>) -> Vec<DfirOp> {
    let mut out = Vec::with_capacity(body.len());
    for op in body {
        let mut clone = op.clone();
        for (from, to) in remap.iter() {
            replace_uses_of_with(&mut clone, *from, *to);
        }
        for region in regions_mut(&mut clone) {
            let original = core::mem::take(region);
            *region = clone_block(&original, vals, remap);
        }
        let olds = results(op);
        for place in results_mut(&mut clone) {
            *place = vals.mint();
        }
        remap.extend(olds.into_iter().zip(results(&clone)));
        out.push(clone);
    }
    out
}

/// `programUnitOp.walk(..)` COLLECTING EVERY `dataflow.get_my_unit_in_collection`, then
/// `op.replaceAllUsesWith(constIntOp.getResult()); op->erase();` (`EnumerateCollectionUnit.cpp:83-90`).
///
/// ⛔ TWO PASSES BECAUSE THE ERASE TAKES THE RESULT WITH IT: the worklist is gathered AS the ops are
/// dropped, and the rewire then runs over what remains. No erased op is itself a user, so the pairwise
/// `replaceAllUsesWith`-then-`erase` and this are the same program.
fn replace_and_erase_my_unit(block: &mut Vec<DfirOp>, index: Val) {
    let mut erased: Vec<Val> = Vec::new();
    collect_and_erase_my_unit(block, &mut erased);
    for from in erased {
        repoint(block, from, index);
    }
}

/// The `workList` of [`replace_and_erase_my_unit`], gathered as the ops are dropped.
fn collect_and_erase_my_unit(block: &mut Vec<DfirOp>, erased: &mut Vec<Val>) {
    block.retain_mut(|op| {
        if let DfirOp::Dataflow(dataflow::Op::GetMyUnitInCollection { result, .. }) = op {
            erased.push(*result);
            return false;
        }
        for region in regions_mut(op) {
            collect_and_erase_my_unit(region, erased);
        }
        true
    });
}

/// `Value::replaceAllUsesWith` over a block and its regions.
fn repoint(block: &mut [DfirOp], from: Val, to: Val) {
    for op in block.iter_mut() {
        replace_uses_of_with(op, from, to);
        for region in regions_mut(op) {
            repoint(region, from, to);
        }
    }
}

/// Replaces: e245_enumerateCollectionUnit
///
/// **245/384** `EnumerateCollectionUnitPass::enumerateCollectionUnit` — `dcc/src/Transform/Dataflow/EnumerateCollectionUnit.cpp:34` (57L).
///
/// ⭐ ONE `dataflow.program_unit` PER MEMBER, each on its own `dataflow.get_unit`, each with the
/// collection body cloned and its `get_my_unit_in_collection` folded to that member's index.
/// ⛔ THE OPS GO WHERE THE COLLECTION STOOD — `OpBuilder builder(collectionDefinitionOp)` (`:52`) —
/// and the collection op itself is erased by the CALLER (entry 286, `:124`).
/// ⛔ THREE OF ITS OPS ARE ABSENT FROM `Dataflow.td` AND THE FILE IS IN `LLVM_OPTIONAL_SOURCES`; they
/// were added to the island for this port — see [`dataflow::Op::GetUnitCollection`].
/// ⛔ AN EMPTY BODY EMITS NOTHING (`:45-49`): a `Vec` region carries no terminator, so the
/// reference's *"body is one op and that op is the terminator"* second clause IS `is_empty`.
#[must_use]
pub fn enumerate_collection_unit(
    collection_definition_op: &dataflow::Op,
    scope: &[DfirOp],
    vals: &mut Values,
) -> Vec<DfirOp> {
    let dataflow::Op::ProgramCollection { unit, body } = collection_definition_op else {
        return Vec::new();
    };
    // `auto type = dyn_cast<VectorType>(collectionDefinitionOp.unit().getType()); if (!type) return;`
    // then `int nUnits = type.getShape()[0];` — the width, and the member list is what carries it.
    let Some(DfirOp::Dataflow(dataflow::Op::GetUnitCollection { members, .. })) =
        defining_op(*unit, scope)
    else {
        return Vec::new();
    };
    if body.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::new();
    // ⛔ A COUNTED ZIP, NOT `enumerate` — `usize -> i64` wants `try_from` and `Err(` is frozen at
    // zero in this crate (`AGENT-BRIEF.md:110`).
    for (i, member) in (0_i64..).zip(members) {
        // `GetUnitOp::create(builder, loc, intType, unitName)` with `name = collectionName + "-" + i`
        // — ⛔ THE NAME IS STRUCTURAL HERE, derived from the residency and the unit by the printer.
        let unit_val = vals.mint();
        out.push(DfirOp::Dataflow(dataflow::Op::GetUnit {
            reg_locale: None,
            result: unit_val,
            residency: member.residency,
            unit: member.unit,
            num_folds: None,
        }));

        // `ConstantIntOp::create(builder, loc, i, intType)` — `intType` is `IntegerType::get(ctx, 32)`
        // (`:53`).
        let index = vals.mint();
        out.push(DfirOp::Arith(arith::Op::ConstantInt {
            result: index,
            value: arith::IntConst::Int { value: i, bits: 32 },
        }));

        let mut unit_body = clone_block(body, vals, &mut Vec::new());
        replace_and_erase_my_unit(&mut unit_body, index);
        out.push(DfirOp::Dataflow(dataflow::Op::ProgramUnit {
            units: vec![unit_val],
            // The reference's `ProgramUnitOp::create` overload takes no precision.
            precision: None,
            body: unit_body,
        }));
    }
    out
}

#[cfg(test)]
mod enumerate_collection_unit_tests {
    use super::*;
    use crate::units::{DfirUnit, Residency};

    /// Two members, a body that reads its own index: each copy gets its own unit, its own literal and
    /// its own values, and no `get_my_unit_in_collection` survives.
    #[test]
    fn each_member_gets_a_program_unit_whose_index_is_a_literal() {
        let mut vals = Values::default();
        let collection = vals.mint();
        let my_unit = vals.mint();
        let sum = vals.mint();
        let scope = vec![DfirOp::Dataflow(dataflow::Op::GetUnitCollection {
            result: collection,
            members: vec![
                dataflow::CollectionMember {
                    residency: Residency::Global,
                    unit: DfirUnit::Hbm,
                },
                dataflow::CollectionMember {
                    residency: Residency::Global,
                    unit: DfirUnit::L3lu,
                },
            ],
        })];
        let program = dataflow::Op::ProgramCollection {
            unit: collection,
            body: vec![
                DfirOp::Dataflow(dataflow::Op::GetMyUnitInCollection {
                    result: my_unit,
                    of: collection,
                }),
                DfirOp::Arith(arith::Op::AddI(arith::IntBinary {
                    result: sum,
                    lhs: my_unit,
                    rhs: my_unit,
                    ty: crate::islands::dataflow_ir::ty::ScalarTy::Int(32),
                })),
            ],
        };

        let out = enumerate_collection_unit(&program, &scope, &mut vals);
        assert_eq!(out.len(), 6);

        for (member, chunk) in [DfirUnit::Hbm, DfirUnit::L3lu]
            .into_iter()
            .zip(out.chunks(3))
        {
            let DfirOp::Dataflow(dataflow::Op::GetUnit { result, unit, .. }) = &chunk[0] else {
                panic!("the member's unit");
            };
            assert_eq!(*unit, member);
            let DfirOp::Arith(arith::Op::ConstantInt { result: index, .. }) = &chunk[1] else {
                panic!("the member's index");
            };
            let DfirOp::Dataflow(dataflow::Op::ProgramUnit { units, body, .. }) = &chunk[2] else {
                panic!("the member's program");
            };
            assert_eq!(units, &vec![*result]);
            // The `get_my_unit_in_collection` is gone and the add reads the literal twice.
            assert_eq!(body.len(), 1);
            let DfirOp::Arith(arith::Op::AddI(add)) = &body[0] else {
                panic!("the cloned add");
            };
            assert_eq!((add.lhs, add.rhs), (*index, *index));
            assert_ne!(add.result, sum);
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 286/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// `moduleOp.walk(..)` COLLECTING EVERY `dataflow.get_total_units_in_collection`, innermost included —
/// the `workList1` of [`run_on_operation`], each paired with the collection it asks about.
fn collect_total_units(scope: &[DfirOp], into: &mut Vec<(Val, Val)>) {
    for op in scope {
        if let DfirOp::Dataflow(dataflow::Op::GetTotalUnitsInCollection { result, of }) = op {
            into.push((*result, *of));
        }
        for region in regions(op) {
            collect_total_units(region, into);
        }
    }
}

/// `OpBuilder builder(op)` THEN `op->erase()` AS ONE WRITE — the constant takes the total-units op's
/// own slot at whatever depth it sat, which is what building *at* it and erasing it amounts to.
fn fold_to_constant(block: &mut [DfirOp], result: Val, count: Val, n_units: i64) {
    for op in block.iter_mut() {
        let is_this_one = matches!(
            op,
            DfirOp::Dataflow(dataflow::Op::GetTotalUnitsInCollection { result: r, .. })
                if *r == result
        );
        if is_this_one {
            *op = DfirOp::Arith(arith::Op::ConstantInt {
                result: count,
                value: arith::IntConst::Int {
                    value: n_units,
                    bits: 32,
                },
            });
            continue;
        }
        for region in regions_mut(op) {
            fold_to_constant(region, result, count, n_units);
        }
    }
}

/// Replaces: e286_runOnOperation
///
/// **286/384** `EnumerateCollectionUnitPass::runOnOperation` — `dcc/src/Transform/Dataflow/EnumerateCollectionUnit.cpp:94` (32L).
///
/// Two rewrites over the module in order: every `get_total_units_in_collection` becomes the member
/// count as an `arith.constant : i32`, then every `dataflow.program_collection` is replaced in place by
/// [`enumerate_collection_unit`]'s expansion of it.
/// ⛔⛔ THE `if (!type) return;` AT `:105` RETURNS FROM THE **PASS**, NOT FROM A WALK LAMBDA — it sits
/// in `runOnOperation`'s own `for`, so one total-units op reading a non-collection abandons every
/// later fold AND the whole collection expansion, with no diagnostic at all.
/// ⛔ THIS FILE IS IN `LLVM_OPTIONAL_SOURCES` (`Transform/Dataflow/CMakeLists.txt:5-6`) and is not
/// built, so there is no vendor case — see [`dataflow::Op::GetUnitCollection`] on its absent ops.
pub fn run_on_operation(module: &mut Vec<DfirOp>, vals: &mut Values) {
    // ── `// Collect the total units operations` (`:97-114`) ──────────────────────────────────────
    let mut total_units: Vec<(Val, Val)> = Vec::new();
    collect_total_units(module, &mut total_units);
    for (result, of) in total_units {
        // `auto type = dyn_cast<VectorType>(op.unit().getType()); if (!type) return;` — the operand is
        // a `vector<Nxindex>` exactly when its definer is the collection binder, and
        // `type.getShape()[0]` is the length of that binder's member list.
        let Some(DfirOp::Dataflow(dataflow::Op::GetUnitCollection { members, .. })) =
            defining_op(of, module)
        else {
            return;
        };
        // ⛔ COUNTED, NOT `try_from` — `Err(` is frozen at zero in this crate (`AGENT-BRIEF.md:110`).
        let n_units = members.iter().fold(0_i64, |n, _| n + 1);
        let count = vals.mint();
        fold_to_constant(module, result, count, n_units);
        // `op.replaceAllUsesWith(constTotalUnitsOp.getResult())` — after the erase, since no erased op
        // is itself a user (entry 245's [`replace_and_erase_my_unit`] makes the same point).
        repoint(module, result, count);
    }

    // ── `// Collect the collection unit operations` (`:116-125`) ─────────────────────────────────
    let collections: Vec<usize> = module
        .iter()
        .enumerate()
        .filter(|(_, op)| matches!(op, DfirOp::Dataflow(dataflow::Op::ProgramCollection { .. })))
        .map(|(at, _)| at)
        .collect();
    // ⛔ DESCENDING, BECAUSE EACH SPLICE SHIFTS EVERY LATER POSITION. The reference holds pointers and
    // has no such problem; `collectionDefinitionOp->erase()` (`:124`) is this splice's removal, and
    // `OpBuilder builder(collectionDefinitionOp)` (entry 245, `:52`) is its insertion.
    for at in collections.into_iter().rev() {
        let collection = module[at].clone();
        let DfirOp::Dataflow(op) = &collection else {
            continue;
        };
        let expanded = enumerate_collection_unit(op, module, vals);
        drop(module.splice(at..=at, expanded));
    }
}

#[cfg(test)]
mod run_on_operation_tests {
    use super::*;
    use crate::units::{DfirUnit, Residency};

    /// 🎯 286/384 — THE TOTAL FOLDS TO THE MEMBER COUNT AND THE COLLECTION EXPANDS WHERE IT STOOD.
    #[test]
    fn the_total_becomes_a_literal_and_the_collection_becomes_one_program_unit_per_member() {
        let mut vals = Values::default();
        let collection = vals.mint();
        let total = vals.mint();
        let member = |unit| dataflow::CollectionMember {
            residency: Residency::Global,
            unit,
        };
        let mut module = vec![
            DfirOp::Dataflow(dataflow::Op::GetUnitCollection {
                result: collection,
                members: vec![member(DfirUnit::Hbm), member(DfirUnit::L3lu)],
            }),
            DfirOp::Dataflow(dataflow::Op::GetTotalUnitsInCollection {
                result: total,
                of: collection,
            }),
            DfirOp::Dataflow(dataflow::Op::ProgramCollection {
                unit: collection,
                body: vec![DfirOp::Dataflow(dataflow::Op::GetMyUnitInCollection {
                    result: vals.mint(),
                    of: collection,
                })],
            }),
        ];

        run_on_operation(&mut module, &mut vals);

        // Neither collection op survives.
        assert!(!module.iter().any(|op| matches!(
            op,
            DfirOp::Dataflow(
                dataflow::Op::GetTotalUnitsInCollection { .. }
                    | dataflow::Op::ProgramCollection { .. }
            )
        )));
        // The member count, then entry 245's one index per member.
        assert_eq!(
            module
                .iter()
                .filter_map(|op| match op {
                    DfirOp::Arith(arith::Op::ConstantInt {
                        value: arith::IntConst::Int { value, bits: 32 },
                        ..
                    }) => Some(*value),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            vec![2, 0, 1]
        );
        assert_eq!(
            module
                .iter()
                .filter(|op| matches!(op, DfirOp::Dataflow(dataflow::Op::ProgramUnit { .. })))
                .count(),
            2
        );
    }
}
