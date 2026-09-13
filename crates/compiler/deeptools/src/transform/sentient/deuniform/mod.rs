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

use crate::arch::Arch;
use crate::islands::dataflow_ir::{Units, ValueMapping, Values};
use crate::islands::sentient::ProgramUnit;
use crate::islands::sentient::dialects::{
    self as dialects, Definitions, LocalRegion, Op, UniformRegions, Val, uniform,
};
use crate::units::DfirUnit;

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

/// Replaces: e436_duplicateAndUpdateEntriesOfQueryMapInLocalRegions
///
/// Every `uniform.query_map` directly inside a local region op's regions gets a narrowed duplicate —
/// mapping only `new_units` — its readers repointed at the duplicate, and its own result queued for
/// deletion.
///
/// ⛔ TRAP: `OpBuilder(Region&)` INSERTS AT THE START OF THE ENTRY BLOCK (`:143`), so each duplicate
/// lands IN FRONT of the query it replaces and the reference's `getOps()` range never revisits it.
/// Appending would make the walk re-see its own output.
///
/// ⛔ `defs` IS THE ENCLOSING SCOPE, supplied by the caller: the mappings a query reads are defined
/// above the local region op, and the region being rewritten cannot be borrowed twice.
pub fn duplicate_and_update_entries_of_query_map_in_local_regions(
    local_region_op: &mut Op,
    new_units: &[Val],
    to_be_deleted: &mut Vec<Val>,
    defs: Definitions<'_>,
    values: &mut Values,
) {
    for region in dialects::regions_mut(local_region_op) {
        let queries: Vec<(Val, Val, Val)> = region
            .iter()
            .filter_map(|op| match op {
                Op::Uniform(uniform::Op::QueryMap { result, map, key }) => {
                    Some((*result, *map, *key))
                }
                _ => None,
            })
            .collect();
        let mut prepended = Vec::new();
        for (result, map, key) in queries {
            let (new_map, new_query) =
                duplicate_and_update_entries_of_query_map(map, key, new_units, defs, values);
            if let Op::Uniform(uniform::Op::QueryMap {
                result: new_result, ..
            }) = new_query
            {
                dialects::replace_all_uses_with(region, result, new_result);
            }
            to_be_deleted.push(result);
            prepended.push(new_map);
            prepended.push(new_query);
        }
        region.splice(0..0, prepended);
    }
}

/// Replaces: e497_duplicateAndUpdateRegionsOfLocalRegionOps
///
/// A copy of a local region op holding only the regions that still run one of `new_units` — each
/// region cloned under a freshly minted argument, its unit list cut to the survivors, and its queries
/// narrowed by [`duplicate_and_update_entries_of_query_map_in_local_regions`].
///
/// ⛔ `nullptr` (`:327`) IS [`None`]: no unit of any region survived, so the caller has nothing to
/// replace the original with.
/// ⛔ `new_to_old_region_num_map` IS THE ORDER ITSELF — the surviving regions are pushed in old-region
/// order, which is what makes the reference's index map an identity on what it keeps.
/// ⛔ TRAP: `getRegIndicesIfExist`/`getRegLocalesIfExist` ARE COPIED WHOLE (`:333`) while the REGION
/// list shrinks. They are per RESULT and the result list is unchanged, so a two-region op cut to one
/// keeps every [`dialects::YieldedReg`] entry — including the discardable `element_sizes` this island
/// fuses into it, which `UniformizeRegionsOp::create` would have dropped.
/// ⛔ THE `DT_CHECK(uniform_op || eq_pattern_op)` (`:291`) IS THE PARAMETER TYPE.
#[must_use]
pub fn duplicate_and_update_regions_of_local_region_ops(
    local_op: &UniformRegions,
    new_units: &[Val],
    to_be_deleted: &mut Vec<Val>,
    defs: Definitions<'_>,
    values: &mut Values,
) -> Option<Op> {
    let mut new_regions: Vec<LocalRegion> = Vec::new();
    for region in local_op.regions() {
        // `expandAllGroupsToUnitsAndUpdateSizes(units_v, list_sizes, builder)` (`:303`) on the
        // reference's own local copies — per region here, for the reason
        // [`expand_all_groups_to_units_and_update_sizes`] records: this island zips the two arrays.
        let mut units = region.units.clone();
        dialects::expand_all_groups_to_units(&mut units, defs);
        let kept: Vec<Val> = units
            .into_iter()
            .filter(|unit| new_units.contains(unit))
            .collect();
        if kept.is_empty() {
            continue;
        }
        // `block.addArgument(builder.getIndexType(), ..)` then `bv_map.map(old args, new args)`
        // (`:336-347`) — the region's own argument is the only value the clone must remap.
        let arg = values.mint();
        let mut mapping = ValueMapping::new();
        mapping.map(region.arg, arg);
        new_regions.push(LocalRegion {
            arg,
            units: kept,
            body: dialects::clone_ops(&region.body, values, &mut mapping),
        });
    }
    if new_regions.is_empty() {
        return None;
    }
    let mut new_op = Op::UniformRegions(match local_op {
        UniformRegions::UniformizeRegions {
            results, yielded, ..
        } => UniformRegions::UniformizeRegions {
            regions: new_regions,
            // `uniform_op.getResultTypes()` — as many results as before, each a new value.
            results: results.iter().map(|_| values.mint()).collect(),
            yielded: yielded.clone(),
        },
        UniformRegions::EqualizePattern { .. } => UniformRegions::EqualizePattern {
            regions: new_regions,
        },
    });
    duplicate_and_update_entries_of_query_map_in_local_regions(
        &mut new_op,
        new_units,
        to_be_deleted,
        defs,
        values,
    );
    Some(new_op)
}

/// Replaces: e556_deuniform
///
/// One program unit rebuilt to run `set_of_units` alone: its body cloned under a fresh argument, then
/// every `uniform.query_map` in it narrowed to those units and every local region op rebuilt without
/// the regions none of them runs — returned with the argument the clone binds.
///
/// ⛔ TRAP: `for (auto fold : unit.getResults())` (`:381-383`) FLATTENS A LIST OF ONE.
/// [`dialects::dataflow::Op::GetUnit`] binds a single value, so the inner loop is the identity here.
/// ⛔ `isa<ReturnOp>` (`:399`) FILTERS NOTHING: this rung's [`ProgramUnit`] body has no terminator.
/// ⛔ THE NEW ARGUMENT IS RETURNED because no op and no [`ProgramUnit`] carries a program unit's
/// `iter_arg` — the same mechanism-from-the-caller e297's `prog_unit_arg` is.
/// ⛔ `DT_CHECK(new_prog_unit.getUnits().size() > 0)` (`:401`) IS [`Units`]: naming no unit is what
/// panics, and it panics where the reference aborts.
#[must_use]
pub fn deuniform<A: Arch>(
    prog_unit: &ProgramUnit<A>,
    prog_unit_arg: Val,
    set_of_units: &[Val],
    enclosing: &[&[Op]],
    values: &mut Values,
) -> (ProgramUnit<A>, Val) {
    let kind = prog_unit.on.kind();
    let bound: Vec<(DfirUnit, Val)> = set_of_units.iter().map(|unit| (kind, *unit)).collect();
    let Some(on) = Units::of(kind, &bound) else {
        panic!("deuniform was handed no unit for the program unit it must rebuild (:401)")
    };
    // `bv_map.map(prog_unit.getBody()->getArguments(), ..)` then `builder.clone(it, bv_map)`
    // (`:396-400`).
    let arg = values.mint();
    let mut mapping = ValueMapping::new();
    mapping.map(prog_unit_arg, arg);
    let mut body = dialects::clone_ops(&prog_unit.body, values, &mut mapping);

    // Every site decided over the clone BEFORE it is touched, because the mappings a query reads and
    // the units a region names are both in it — see e297's own note on the same ordering.
    let mut edits: Vec<(Vec<(usize, usize)>, usize, Vec<Op>)> = Vec::new();
    let mut rewires: Vec<(Val, Val)> = Vec::new();
    let mut to_be_deleted: Vec<Val> = Vec::new();
    {
        let mut scope: Vec<&[Op]> = vec![body.as_slice()];
        scope.extend_from_slice(enclosing);
        let defs = Definitions::within_program_unit(&scope, arg, set_of_units);
        for (path, at) in deuniform_sites(&body) {
            let Some(op) = block_at(&body, &path).and_then(|block| block.get(at)) else {
                continue;
            };
            match op {
                Op::Uniform(uniform::Op::QueryMap { result, map, key }) => {
                    let (new_map, new_query) = duplicate_and_update_entries_of_query_map(
                        *map,
                        *key,
                        set_of_units,
                        defs,
                        values,
                    );
                    if let Op::Uniform(uniform::Op::QueryMap {
                        result: new_result, ..
                    }) = new_query
                    {
                        rewires.push((*result, new_result));
                    }
                    // `for (auto user : map_op->getUsers()) num_users++; if (num_users <= 1)`
                    // (`:404-406`) — this query is that one user, and a map two queries read survives
                    // both of them, as it does there.
                    if dialects::use_count(*map, &body) <= 1 {
                        to_be_deleted.push(*map);
                    }
                    edits.push((path, at, vec![new_map, new_query]));
                }
                Op::UniformRegions(local) => {
                    if let Some(new_op) = duplicate_and_update_regions_of_local_region_ops(
                        local,
                        set_of_units,
                        &mut to_be_deleted,
                        defs,
                        values,
                    ) {
                        rewires.extend(
                            dialects::results(op)
                                .into_iter()
                                .zip(dialects::results(&new_op)),
                        );
                        edits.push((path, at, vec![new_op]));
                    }
                }
                _ => {}
            }
        }
    }
    // ⭐ IN REVERSE DOCUMENT ORDER so an edit cannot move a site still to be applied. The reference
    // inserts after the old op and erases it last (`:445`); replacing it is the same net shape.
    for (path, at, replacement) in edits.into_iter().rev() {
        if let Some(block) = block_at_mut(&mut body, &path) {
            block.splice(at..=at, replacement);
        }
    }
    for (of, with) in rewires {
        dialects::replace_all_uses_with(&mut body, of, with);
    }
    // `for (auto op : to_be_deleted) op->erase();` (`:445`) — including what e497 queued inside the
    // regions it rebuilt.
    for val in to_be_deleted {
        dialects::erase_defining_op(&mut body, val);
    }

    (
        ProgramUnit {
            on,
            precision: prog_unit.precision,
            body,
            arch: core::marker::PhantomData,
        },
        arg,
    )
}

/// e556's `walk<WalkOrder::PreOrder>` (`:403`), as the address of each site: the path of
/// (op index, region index) pairs to the block that holds it, then the op's index in that block.
///
/// ⛔ A LOCAL REGION OP'S OWN SUBTREE IS **SKIPPED** (`:429`, `:439`) — which is why e436 exists to
/// rewrite the queries inside the copy e497 builds.
/// ⛔ THE RUNG BELOW'S SPELLING IS A `todo!`: [`uniform::Op::UniformizeRegions`] holds a DataflowIR
/// body, and e497 rebuilds THIS rung's [`UniformRegions`] — the two are different types, not two
/// names for one op.
fn deuniform_sites(block: &[Op]) -> Vec<(Vec<(usize, usize)>, usize)> {
    fn walk(
        block: &[Op],
        path: &mut Vec<(usize, usize)>,
        found: &mut Vec<(Vec<(usize, usize)>, usize)>,
    ) {
        for (at, op) in block.iter().enumerate() {
            match op {
                Op::Uniform(uniform::Op::QueryMap { .. }) => found.push((path.clone(), at)),
                Op::UniformRegions(_) => {
                    found.push((path.clone(), at));
                    continue;
                }
                Op::Uniform(
                    uniform::Op::UniformizeRegions { .. } | uniform::Op::EqualizePattern { .. },
                ) => todo!(
                    "uniform::UniformizeRegionsOp/EqualizePatternOp carrying a DataflowIR body — \
                     raising it to this rung's uniform.uniformize_regions is out of campaign scope"
                ),
                _ => {}
            }
            for (region_num, region) in dialects::regions_ref(op).into_iter().enumerate() {
                path.push((at, region_num));
                walk(region, path, found);
                path.pop();
            }
        }
    }
    let mut found = Vec::new();
    walk(block, &mut Vec::new(), &mut found);
    found
}

/// The block a [`deuniform_sites`] path addresses.
fn block_at<'b>(block: &'b [Op], path: &[(usize, usize)]) -> Option<&'b [Op]> {
    let Some((&(at, region_num), rest)) = path.split_first() else {
        return Some(block);
    };
    let inner = *dialects::regions_ref(block.get(at)?).get(region_num)?;
    block_at(inner, rest)
}

/// The same block, FOR THE EDIT.
fn block_at_mut<'b>(
    block: &'b mut Vec<Op>,
    path: &[(usize, usize)],
) -> Option<&'b mut Vec<Op>> {
    let Some((&(at, region_num), rest)) = path.split_first() else {
        return Some(block);
    };
    let inner = dialects::regions_mut(block.get_mut(at)?)
        .into_iter()
        .nth(region_num)?;
    block_at_mut(inner, rest)
}

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
    use crate::arch::Dd2;
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

    /// e497 — the region whose units the new unit set no longer covers is dropped, the surviving one
    /// is cloned under a fresh argument over just those units with its query narrowed, and a set that
    /// covers nothing answers the reference's `nullptr`.
    #[test]
    fn duplicating_a_local_region_op_keeps_only_the_regions_the_new_units_still_run() {
        let func_body = vec![
            get_unit(101),
            get_unit(102),
            get_unit(103),
            group(104, &[101, 102]),
            Op::Uniform(uniform::Op::DefImmutableMapping {
                result: Val(105),
                pairs: vec![
                    (Val(101), Val(120)),
                    (Val(102), Val(121)),
                    (Val(103), Val(122)),
                ],
            }),
        ];
        let scopes: Vec<&[Op]> = vec![&func_body];
        let defs = Definitions::within_program_unit(
            &scopes,
            Val(109),
            &[Val(101), Val(102), Val(103)],
        );
        let local_op = UniformRegions::UniformizeRegions {
            regions: vec![
                region(
                    110,
                    &[104],
                    vec![
                        Op::Uniform(uniform::Op::QueryMap {
                            result: Val(141),
                            map: Val(105),
                            key: Val(109),
                        }),
                        Op::Uniform(uniform::Op::Yield {
                            operands: vec![Val(141)],
                        }),
                    ],
                ),
                region(111, &[103], Vec::new()),
            ],
            results: vec![Val(130)],
            yielded: Vec::new(),
        };
        let mut values = Values::default();
        let mut to_be_deleted = Vec::new();

        let duplicated = duplicate_and_update_regions_of_local_region_ops(
            &local_op,
            &[Val(102)],
            &mut to_be_deleted,
            defs,
            &mut values,
        );

        let Some(Op::UniformRegions(new_op)) = &duplicated else {
            panic!("expected a duplicated uniformize_regions, got {duplicated:?}");
        };
        assert_eq!(new_op.regions().len(), 1, "region 1 ran only unit 103");
        // The group expanded, and only the unit the caller kept survived it.
        assert_eq!(new_op.regions()[0].units, vec![Val(102)]);
        assert_eq!(new_op.regions()[0].arg, Val(0), "a freshly minted argument");
        assert_eq!(dialects::results(&duplicated.clone().expect("built")), vec![Val(2)]);
        // e436's narrowed duplicate sits in front of the clone, and the yield reads it.
        assert_eq!(
            new_op.regions()[0].body,
            vec![
                Op::Uniform(uniform::Op::DefImmutableMapping {
                    result: Val(3),
                    pairs: vec![(Val(102), Val(121))],
                }),
                Op::Uniform(uniform::Op::QueryMap {
                    result: Val(4),
                    map: Val(3),
                    key: Val(109),
                }),
                Op::Uniform(uniform::Op::QueryMap {
                    result: Val(1),
                    map: Val(105),
                    key: Val(109),
                }),
                Op::Uniform(uniform::Op::Yield {
                    operands: vec![Val(4)],
                }),
            ]
        );
        assert_eq!(to_be_deleted, vec![Val(1)]);

        // `if (new_local_units.size() == 0) return nullptr;`
        let mut fresh = Values::default();
        assert!(
            duplicate_and_update_regions_of_local_region_ops(
                &local_op,
                &[Val(199)],
                &mut Vec::new(),
                defs,
                &mut fresh,
            )
            .is_none()
        );
    }

    /// e436 — the query inside the local region gains a narrowed duplicate IN FRONT of itself, its
    /// reader moves onto the duplicate, and the query it replaces is queued for deletion.
    #[test]
    fn a_local_region_query_map_is_duplicated_ahead_of_itself() {
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
        let mut op = Op::UniformRegions(UniformRegions::UniformizeRegions {
            regions: vec![region(
                11,
                &[2],
                vec![
                    Op::Uniform(uniform::Op::QueryMap {
                        result: Val(41),
                        map: Val(10),
                        key: Val(9),
                    }),
                    copy(42, 41),
                ],
            )],
            results: Vec::new(),
            yielded: Vec::new(),
        });
        let mut to_be_deleted = Vec::new();

        duplicate_and_update_entries_of_query_map_in_local_regions(
            &mut op,
            &[Val(2)],
            &mut to_be_deleted,
            defs,
            &mut values,
        );

        let Op::UniformRegions(rewritten) = &op else {
            panic!("expected a uniformize_regions, got {op:?}");
        };
        let body = &rewritten.regions()[0].body;
        assert_eq!(body.len(), 4);
        assert_eq!(
            body[0],
            Op::Uniform(uniform::Op::DefImmutableMapping {
                result: Val(0),
                pairs: vec![(Val(2), Val(21))],
            })
        );
        assert_eq!(
            body[1],
            Op::Uniform(uniform::Op::QueryMap {
                result: Val(1),
                map: Val(0),
                key: Val(9),
            })
        );
        assert_eq!(copies(body), vec![(42, 1)]);
        assert_eq!(to_be_deleted, vec![Val(41)]);
    }

    /// e556: the unit is rebuilt for two of its three units — the query is replaced by one over a
    /// mapping narrowed to those two, its reader follows, and the wide mapping nothing reads any more
    /// is gone.
    #[test]
    fn deuniform_narrows_the_query_map_to_the_units_the_new_unit_runs() {
        let func_body = vec![get_unit(1), get_unit(2), get_unit(3)];
        let prog_unit: ProgramUnit<Dd2> = ProgramUnit {
            on: Units::of(
                DfirUnit::Lxlu,
                &[
                    (DfirUnit::Lxlu, Val(1)),
                    (DfirUnit::Lxlu, Val(2)),
                    (DfirUnit::Lxlu, Val(3)),
                ],
            )
            .expect("three units of one kind"),
            precision: None,
            body: vec![
                Op::Uniform(uniform::Op::DefImmutableMapping {
                    result: Val(10),
                    pairs: vec![(Val(1), Val(20)), (Val(2), Val(21)), (Val(3), Val(22))],
                }),
                Op::Uniform(uniform::Op::QueryMap {
                    result: Val(11),
                    map: Val(10),
                    key: Val(9),
                }),
                copy(12, 11),
            ],
            arch: core::marker::PhantomData,
        };
        let mut values = Values::default();
        // Past every name the fixture writes, so a minted clone cannot collide with a unit value.
        while values.issued() <= 22 {
            values.mint();
        }

        let (new_unit, arg) = deuniform(
            &prog_unit,
            Val(9),
            &[Val(1), Val(2)],
            &[&func_body],
            &mut values,
        );

        assert_eq!(new_unit.on.vals(), vec![Val(1), Val(2)]);
        // The wide mapping was erased with the query that was its only reader.
        assert_eq!(new_unit.body.len(), 3);
        let Op::Uniform(uniform::Op::DefImmutableMapping { result, pairs }) = &new_unit.body[0]
        else {
            panic!("expected the narrowed mapping, got {:?}", new_unit.body[0]);
        };
        assert_eq!(*pairs, vec![(Val(1), Val(20)), (Val(2), Val(21))]);
        let Op::Uniform(uniform::Op::QueryMap {
            result: query,
            map,
            key,
        }) = &new_unit.body[1]
        else {
            panic!("expected the narrowed query, got {:?}", new_unit.body[1]);
        };
        assert_eq!((*map, *key), (*result, arg));
        let readers = copies(&new_unit.body);
        assert_eq!(readers.len(), 1);
        assert_eq!(readers[0].1, query.0);
    }
}
