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

//! `Deuniform.cpp` — 9 of the campaign's 656 units (dependency level(s) [0, 1, 2, 3, 4, 5, 6]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e039_existsInCollection` | 039 | 0 | 8 | `dcc/src/Transform/Sentient/Deuniform.cpp:128` |
//! | `e297_simplifyProgramUnitOp` | 297 | 1 | 64 | `dcc/src/Transform/Sentient/Deuniform.cpp:155` |
//! | `e298_duplicateAndUpdateEntriesOfQueryMap` | 298 | 1 | 22 | `dcc/src/Transform/Sentient/Deuniform.cpp:221` |
//! | `e299_expandAllGroupsToUnitsAndUpdateSizes` | 299 | 1 | 35 | `dcc/src/Transform/Sentient/Deuniform.cpp:246` |
//! | `e436_duplicateAndUpdateEntriesOfQueryMapInLocalRegions` | 436 | 2 | 13 | `dcc/src/Transform/Sentient/Deuniform.cpp:139` |
//! | `e497_duplicateAndUpdateRegionsOfLocalRegionOps` | 497 | 3 | 88 | `dcc/src/Transform/Sentient/Deuniform.cpp:284` |
//! | `e556_deuniform` | 556 | 4 | 70 | `dcc/src/Transform/Sentient/Deuniform.cpp:375` |
//! | `e595_deuniform` | 595 | 5 | 30 | `dcc/src/Transform/Sentient/Deuniform.cpp:448` |
//! | `e621_runOnOperation` | 621 | 6 | 42 | `dcc/src/Transform/Sentient/Deuniform.cpp:482` |

// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET, so every item below is reachable only from this
// file's own tests until `e621_runOnOperation` (level 6) lands and something calls it. CI runs clippy
// with `-D warnings`, so without this the first ported leaf of a 9-unit module fails the gate.
// ⭐ REMOVE THIS WITH e621: at that point an unused item here is a real defect again.
#![allow(dead_code)]

use crate::islands::dataflow_ir::Values;
use crate::islands::sentient::dialects::{self as dialects, Definitions, Op, Val, uniform};

/// Replaces: e039_existsInCollection
///
/// Whether this unit is already a member of one of the collected sets of units.
///
/// TRAP: the identity is the `Val` the `dataflow.get_unit` BINDS — every `unit_op` this pass compares
/// arrives through `unit.getDefiningOp()` (`dcc/src/Transform/Sentient/Deuniform.cpp:505-506`), so equal
/// values are the same op.
pub fn exists_in_collection(unit_op: Val, collection: &[Vec<Val>]) -> bool {
    collection.iter().any(|set| set.contains(&unit_op))
}

/// Replaces: e297_simplifyProgramUnitOp
///
/// A deuniformed unit loses the local regions that no longer say anything: one over units this unit
/// does not run is deleted, one over exactly this unit's units is spliced into the body, a
/// `uniform.query_map` on the unit's own argument with a single answer becomes that answer, and a
/// `uniform.def_immutable_mapping` nothing reads any more is dropped.
///
/// ⛔ TRAP: THE REFERENCE'S FIRST ERASE LOOP IS A LATENT USE-AFTER-FREE. `Operation::walk` defaults
/// to `WalkOrder::PostOrder`, so `to_be_deleted` holds inner local regions BEFORE their parents and
/// `for (int i = size - 1; i >= 0; i--)` (`:213`) erases a parent before the child it already freed.
/// This port recurses into regions first and then walks each block downward, which is the same net
/// effect on the shape the reference survives.
///
/// ⛔ `prog_unit_arg` AND `enclosing` ARE THE DROPPED MECHANISM: no op of this island carries a
/// program unit's `iter_arg` or a parent pointer, and the `dataflow.get_unit`s in `units` are defined
/// ABOVE the unit — see [`Definitions::within_program_unit`].
pub fn simplify_program_unit_op(
    unit_body: &mut Vec<Op>,
    units: &[Val],
    prog_unit_arg: Val,
    enclosing: &[&[Op]],
) {
    simplify_local_regions(
        unit_body,
        units,
        prog_unit_arg,
        Definitions::from_innermost(enclosing),
    );

    // `query_op.replaceAllUsesWith(values[0])` (`:288`) — decided over an immutable body, because the
    // mapping the query reads is IN that body.
    let mut answered: Vec<(Val, Val)> = Vec::new();
    {
        let mut scope: Vec<&[Op]> = vec![unit_body.as_slice()];
        scope.extend_from_slice(enclosing);
        let defs = Definitions::within_program_unit(&scope, prog_unit_arg, units);
        for op in collect_uniform(unit_body) {
            let uniform::Op::QueryMap { result, map, key } = op else {
                continue;
            };
            let values = dialects::uniform_mapping_values(map, key, defs);
            if values.len() == 1
                && key == prog_unit_arg
                && dialects::uniform_mapping_keys(key, defs).len() <= units.len()
            {
                answered.push((result, values[0]));
            }
        }
    }
    for (result, value) in answered {
        dialects::replace_all_uses_with(unit_body, result, value);
        dialects::erase_defining_op(unit_body, result);
    }

    // `if (mapping_op.use_empty())` (`:295`) — ⭐ EXACT RATHER THAN APPROXIMATE over the unit's body:
    // a value defined inside a region has no readers outside it.
    let unused: Vec<Val> = collect_uniform(unit_body)
        .into_iter()
        .filter_map(|op| match op {
            uniform::Op::DefImmutableMapping { result, .. } => Some(result),
            _ => None,
        })
        .filter(|result| dialects::use_count(*result, unit_body) == 0)
        .collect();
    for result in unused {
        dialects::erase_defining_op(unit_body, result);
    }
}

/// The first walk of e297 (`:161-212`), one block at a time: regions first, then the block's own ops
/// downward so a removal cannot invalidate an index still to be visited.
fn simplify_local_regions(
    block: &mut Vec<Op>,
    units: &[Val],
    prog_unit_arg: Val,
    defs: Definitions<'_>,
) {
    for op in block.iter_mut() {
        for region in dialects::regions_mut(op) {
            simplify_local_regions(region, units, prog_unit_arg, defs);
        }
    }
    for at in (0..block.len()).rev() {
        let Some(regions) = dialects::local_region_count(&block[at]) else {
            continue;
        };
        let mut local_units = dialects::operands(&block[at]);
        dialects::expand_all_groups_to_units(&mut local_units, defs);
        if !local_units.iter().any(|unit| units.contains(unit)) {
            block.remove(at);
        } else if regions == 1
            && local_units.len() == units.len()
            && local_units.iter().all(|unit| units.contains(unit))
        {
            dialects::extract_op_from_local_region(block, at, prog_unit_arg);
        }
    }
}

/// Every `uniform.query_map` and `uniform.def_immutable_mapping` under `scope`, in walk order.
///
/// ⛔ A LOWER-RUNG LOCAL REGION IS NOT DESCENDED INTO, as everywhere in this island: its body holds
/// ops of the rung below. e297's two walks only ever act on a query whose key is the program unit's
/// own argument, and the only way one comes to be nested is
/// [`dialects::extract_op_from_local_region`], which raises the body it splices.
fn collect_uniform(scope: &[Op]) -> Vec<uniform::Op> {
    let mut found = Vec::new();
    fn walk(scope: &[Op], found: &mut Vec<uniform::Op>) {
        for op in scope {
            if let Op::Uniform(
                inner @ (uniform::Op::QueryMap { .. } | uniform::Op::DefImmutableMapping { .. }),
            ) = op
            {
                found.push(inner.clone());
            }
            for region in dialects::regions_ref(op) {
                walk(region, found);
            }
        }
    }
    walk(scope, &mut found);
    found
}

/// Replaces: e298_duplicateAndUpdateEntriesOfQueryMap
///
/// A fresh `uniform.def_immutable_mapping` holding only the pairs whose key is one of `new_units`,
/// and a fresh `uniform.query_map` reading it with the same key — returned in creation order.
///
/// ⛔ TRAP: THE REFERENCE PAIRS BY THE **KEY** INDEX INTO A VALUE LIST THAT DROPPED ITS MISSES.
/// `values[i]` (`:234`) is read at the key's position while `getListOfValueOpsFromUniformMapping`
/// pushes only the keys the mapping holds (`Uniform.cpp:548-556`), so one absent key shifts every
/// later pair and the last read runs off the end. Zipping reproduces the shift for the pairs that
/// exist and stops where the reference reads out of bounds.
pub fn duplicate_and_update_entries_of_query_map(
    map: Val,
    key: Val,
    new_units: &[Val],
    defs: Definitions<'_>,
    values: &mut Values,
) -> (Op, Op) {
    let keys = dialects::uniform_mapping_keys(key, defs);
    let pairs: Vec<(Val, Val)> = keys
        .iter()
        .zip(dialects::uniform_mapping_values(map, key, defs))
        .filter(|(mapped, _)| new_units.contains(mapped))
        .map(|(mapped, value)| (*mapped, value))
        .collect();
    let new_map = values.mint();
    (
        Op::Uniform(uniform::Op::DefImmutableMapping {
            result: new_map,
            pairs,
        }),
        Op::Uniform(uniform::Op::QueryMap {
            result: values.mint(),
            map: new_map,
            key,
        }),
    )
}

/// Replaces: e299_expandAllGroupsToUnitsAndUpdateSizes
///
/// Every `dataflow.create_group` in a local region op's unit lists replaced by the group's members.
///
/// ⛔ TRAP: `list_sizes` CANNOT DESYNC FROM `$units` HERE, so the reference's
/// `DT_CHECK(list_sizes[s] == 1)` (`:263`) has nothing to check: [`dialects::LocalRegion`] zips the
/// two, and a region's size IS its unit count. A region of more than one unit holding a group — the
/// case the check forbids — gets the generalization the zip forces, not the reference's two arrays of
/// different lengths.
///
/// ⛔ TRAP: THE REFERENCE'S FIRST LOOP (`:251-258`) IS DEAD — it fills a local `units_in_groups` per
/// iteration and drops it.
pub fn expand_all_groups_to_units_and_update_sizes(op: &mut Op, defs: Definitions<'_>) {
    match op {
        Op::UniformRegions(regions) => {
            for region in regions.regions_mut() {
                dialects::expand_all_groups_to_units(&mut region.units, defs);
            }
        }
        Op::Uniform(
            uniform::Op::UniformizeRegions { regions, .. }
            | uniform::Op::EqualizePattern { regions },
        ) => {
            for region in regions.iter_mut() {
                dialects::expand_all_groups_to_units(&mut region.units, defs);
            }
        }
        // Nothing else carries a `$units`/`$list_sizes` pair.
        _ => {}
    }
}

// crustify:todo: e436_duplicateAndUpdateEntriesOfQueryMapInLocalRegions
//   authority : dcc/src/Transform/Sentient/Deuniform.cpp:139  (13 body lines, level 2)
//   original  : void DeuniformPass::duplicateAndUpdateEntriesOfQueryMapInLocalRegions( mlir::Operation *local_region_op, std::vector<mlir::Value> &new_units, std::vector<mlir::Operation *> &to_be_deleted)
//   calls     : e298_duplicateAndUpdateEntriesOfQueryMap

// crustify:todo: e497_duplicateAndUpdateRegionsOfLocalRegionOps
//   authority : dcc/src/Transform/Sentient/Deuniform.cpp:284  (88 body lines, level 3)
//   original  : mlir::Operation *DeuniformPass::duplicateAndUpdateRegionsOfLocalRegionOps( mlir::Operation *local_op, std::vector<mlir::Value> &new_units, mlir::OpBuilder builder, std::vector<mlir::Operation *> &to_be_deleted)
//   calls     : e252_size, e299_expandAllGroupsToUnitsAndUpdateSizes, e436_duplicateAndUpdateEntriesOfQueryMapInLocalRegions

// crustify:todo: e556_deuniform
//   authority : dcc/src/Transform/Sentient/Deuniform.cpp:375  (70 body lines, level 4)
//   original  : mlir::dataflow::ProgramUnitOp DeuniformPass::deuniform( mlir::dataflow::ProgramUnitOp prog_unit, std::vector<mlir::dataflow::GetUnitOp> &set_of_units)
//   calls     : e252_size, e298_duplicateAndUpdateEntriesOfQueryMap, e497_duplicateAndUpdateRegionsOfLocalRegionOps

// crustify:todo: e595_deuniform
//   authority : dcc/src/Transform/Sentient/Deuniform.cpp:448  (30 body lines, level 5)
//   original  : std::vector<mlir::dataflow::ProgramUnitOp> DeuniformPass::deuniform( mlir::dataflow::ProgramUnitOp prog_unit, std::vector<std::vector<mlir::dataflow::GetUnitOp>> &set_of_unit_collection)
//   calls     : e252_size, e556_deuniform

// crustify:todo: e621_runOnOperation
//   authority : dcc/src/Transform/Sentient/Deuniform.cpp:482  (42 body lines, level 6)
//   original  : void DeuniformPass::runOnOperation()
//   calls     : e039_existsInCollection, e252_size, e297_simplifyProgramUnitOp, e556_deuniform, e595_deuniform

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::islands::dataflow_ir::dialects::dataflow;
    use crate::islands::sentient::dialects::sentient::{Reg, RegType};
    use crate::islands::sentient::dialects::{LocalRegion, UniformRegions, sentient};
    use crate::units::{DfirUnit, Residency};

    /// `%r = dataflow.get_unit {name, type} : index`.
    fn get_unit(result: u32) -> Op {
        Op::Dataflow(dataflow::Op::GetUnit {
            result: Val(result),
            residency: Residency::Global,
            unit: DfirUnit::Lxlu,
            num_folds: None,
        })
    }

    /// `%r = dataflow.create_group(..) : index`.
    fn group(result: u32, members: &[u32]) -> Op {
        Op::Dataflow(dataflow::Op::CreateGroup {
            result: Val(result),
            unit_ids: members.iter().map(|m| Val(*m)).collect(),
        })
    }

    /// `%r = sentient.scalar_copy %i` — a reader, so a rewire is observable.
    fn copy(result: u32, input: u32) -> Op {
        Op::Sentient(sentient::Op::ScalarCopy {
            input: Val(input),
            result: Val(result),
            reg: Reg {
                locale: RegType::Unknown,
                index: None,
            },
            element_size: None,
            program_header: false,
        })
    }

    /// One region of a local region op — its argument, its units and its body.
    fn region(arg: u32, units: &[u32], body: Vec<Op>) -> LocalRegion {
        LocalRegion {
            arg: Val(arg),
            units: units.iter().map(|u| Val(*u)).collect(),
            body,
        }
    }

    /// Every `sentient.scalar_copy` in a block, as `(result, input)`.
    fn copies(scope: &[Op]) -> Vec<(u32, u32)> {
        scope
            .iter()
            .filter_map(|op| match op {
                Op::Sentient(sentient::Op::ScalarCopy { input, result, .. }) => {
                    Some((result.0, input.0))
                }
                _ => None,
            })
            .collect()
    }

    /// A unit in the second set is found; one in no set is not.
    #[test]
    fn exists_in_collection_searches_every_set() {
        let collection = vec![vec![Val(1), Val(2)], vec![Val(3)]];

        assert!(exists_in_collection(Val(3), &collection));
        assert!(!exists_in_collection(Val(4), &collection));
    }

    /// All four simplifications on one body: the foreign `equalize_pattern` goes, the
    /// `uniformize_regions` over exactly this unit's units is spliced in with both its rewires, the
    /// query on the unit's own argument becomes the one value it can answer, and the mapping that
    /// query was the only reader of goes with it.
    #[test]
    fn a_deuniformed_unit_loses_every_local_region_that_says_nothing() {
        let func_body = vec![get_unit(1), get_unit(2), get_unit(3), group(4, &[1, 2])];
        let units = vec![Val(1), Val(2)];
        let foreign = Op::UniformRegions(UniformRegions::EqualizePattern {
            regions: vec![region(
                11,
                &[3],
                vec![Op::Uniform(uniform::Op::Yield {
                    operands: Vec::new(),
                })],
            )],
        });
        let whole = Op::UniformRegions(UniformRegions::UniformizeRegions {
            regions: vec![region(
                10,
                &[4],
                vec![
                    copy(20, 10),
                    Op::Uniform(uniform::Op::Yield {
                        operands: vec![Val(20)],
                    }),
                ],
            )],
            results: vec![Val(30)],
            yielded: Vec::new(),
        });
        let mut unit_body = vec![
            foreign,
            whole,
            copy(31, 30),
            Op::Uniform(uniform::Op::DefImmutableMapping {
                result: Val(40),
                pairs: vec![(Val(1), Val(50))],
            }),
            Op::Uniform(uniform::Op::QueryMap {
                result: Val(41),
                map: Val(40),
                key: Val(9),
            }),
            copy(42, 41),
        ];

        simplify_program_unit_op(&mut unit_body, &units, Val(9), &[&func_body]);

        assert_eq!(unit_body.len(), 3);
        // `%20` now reads the program unit's own argument, `%31` the value the region yielded, and
        // `%42` the mapping's one answer.
        assert_eq!(copies(&unit_body), vec![(20, 9), (31, 20), (42, 50)]);
    }

    /// The duplicate keeps only the pairs whose key is one of the new units, and the fresh query
    /// reads the fresh mapping.
    #[test]
    fn a_duplicated_query_map_drops_the_units_it_no_longer_covers() {
        let scope = vec![
            get_unit(1),
            get_unit(2),
            Op::Uniform(uniform::Op::DefImmutableMapping {
                result: Val(10),
                pairs: vec![(Val(1), Val(20)), (Val(2), Val(21))],
            }),
        ];
        let regions: Vec<&[Op]> = vec![&scope];
        let defs = Definitions::within_program_unit(&regions, Val(9), &[Val(1), Val(2)]);
        let mut values = Values::default();

        let (map, query) = duplicate_and_update_entries_of_query_map(
            Val(10),
            Val(9),
            &[Val(2)],
            defs,
            &mut values,
        );

        let Op::Uniform(uniform::Op::DefImmutableMapping { result, pairs }) = &map else {
            panic!("expected a def_immutable_mapping, got {map:?}");
        };
        assert_eq!(*pairs, vec![(Val(2), Val(21))]);
        assert_eq!(
            query,
            Op::Uniform(uniform::Op::QueryMap {
                result: Val(1),
                map: *result,
                key: Val(9),
            })
        );
    }

    /// The grouped region's units become the group's members; the region that names a unit outright
    /// is untouched.
    #[test]
    fn expanding_groups_rewrites_only_the_regions_that_hold_one() {
        let func_body = vec![get_unit(1), get_unit(2), get_unit(3), group(4, &[1, 2])];
        let regions: Vec<&[Op]> = vec![&func_body];
        let mut op = Op::UniformRegions(UniformRegions::UniformizeRegions {
            regions: vec![region(10, &[4], Vec::new()), region(11, &[3], Vec::new())],
            results: Vec::new(),
            yielded: Vec::new(),
        });

        expand_all_groups_to_units_and_update_sizes(&mut op, Definitions::from_innermost(&regions));

        let Op::UniformRegions(rewritten) = &op else {
            panic!("expected a uniformize_regions, got {op:?}");
        };
        assert_eq!(rewritten.regions()[0].units, vec![Val(1), Val(2)]);
        assert_eq!(rewritten.regions()[1].units, vec![Val(3)]);
    }
}
