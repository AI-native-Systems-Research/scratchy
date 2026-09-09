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
use crate::islands::dataflow_ir::Values;
use crate::islands::sentient::ProgramUnit;
use crate::islands::sentient::dialects::{self as dialects, Definitions, Op, Val, dataflow, uniform};

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
/// Drops every local region one `dataflow.program_unit`'s units have nothing in common with, flattens
/// the one whose units are exactly theirs, folds the query maps that can then answer only once, and
/// erases the mappings left unread.
///
/// TRAP: `Operation::walk` defaults to `WalkOrder::PostOrder`, so the reference's reverse-order erase
/// is OUTERMOST-first and an outer erase takes its inner marks with it — which is what recursing
/// before deciding reproduces. TRAP: a query map inside a LOWER-rung local region is not reached, for
/// the reason [`dialects::regions_ref`] records.
pub(crate) fn simplify_program_unit_op<A: Arch>(unit: &mut ProgramUnit<A>, preamble: &[Op]) {
    let Some(prog_arg) = unit.iter_arg else {
        todo!(
            "simplifyProgramUnitOp: the program unit binds no iter_arg, so \
             `prog_unit_op.getBody()->getArgument(0)` (Deuniform.cpp:157) is out of bounds"
        );
    };
    let units = unit.on.vals();

    // ⭐ THE PREAMBLE IS THE WHOLE SCOPE PHASE ONE NEEDS, and provably so: a local region's units are
    // OPERANDS of an op inside the program unit, so their `dataflow.get_unit`/`dataflow.create_group`
    // declarations are necessarily outside it. That is what lets the walk hold `unit.body` mutably.
    let outer: [&[Op]; 1] = [preamble];
    let rewires = simplify_local_ops(
        &mut unit.body,
        &units,
        prog_arg,
        Definitions::from_innermost(&outer),
    );
    for (of, with) in rewires {
        dialects::replace_all_uses_with(&mut unit.body, of, with);
    }

    // The second walk reads a body the first has already rewritten (`Deuniform.cpp:210-221`), so its
    // `Definitions` is built against that body and dropped before anything is erased.
    let folds = {
        let inner: [&[Op]; 2] = [&unit.body, preamble];
        let defs = Definitions::from_innermost(&inner).with_program_unit(prog_arg, &units);
        query_map_folds(&unit.body, prog_arg, &units, defs)
    };
    for (result, answer) in folds {
        dialects::replace_all_uses_with(&mut unit.body, result, answer);
        dialects::erase_defining_op(&mut unit.body, result);
    }

    for result in unread_mappings(&unit.body) {
        dialects::erase_defining_op(&mut unit.body, result);
    }
}

/// What the first walk of [`simplify_program_unit_op`] decides for one `uniform` local-region op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LocalOp {
    /// `to_be_deleted.push_back(local_op)` — no unit in common with the program unit's own.
    Delete,
    /// `extractOpFromLocalRegion` and then delete — its one region runs on exactly those units.
    Extract,
}

/// The first walk and its erase (`Deuniform.cpp:158-208`), innermost regions first, answering the
/// `uniform_op.getResult(k)` rewires `extractOpFromLocalRegion` leaves behind.
///
/// ⛔ THE REWIRES ARE THE CALLER'S TO APPLY, because `replaceAllUsesWith` is not region-local: a
/// result yielded out of a local region can be read anywhere in the program unit.
fn simplify_local_ops(
    body: &mut Vec<Op>,
    units: &[Val],
    prog_arg: Val,
    defs: Definitions<'_>,
) -> Vec<(Val, Val)> {
    let mut rewires = Vec::new();
    for op in body.iter_mut() {
        for region in dialects::regions_mut(op) {
            rewires.extend(simplify_local_ops(region, units, prog_arg, defs));
        }
    }
    let mut at = 0;
    while at < body.len() {
        match local_op_action(&body[at], units, defs) {
            None => at += 1,
            Some(LocalOp::Delete) => {
                body.remove(at);
            }
            Some(LocalOp::Extract) => {
                let (moved, extracted) = extract_op_from_local_region(body, at, prog_arg);
                rewires.extend(extracted);
                at += moved;
            }
        }
    }
    rewires
}

/// The walk callback's decision — `None` for any op that is not a local-region op, and for one whose
/// units overlap the program unit's without matching them.
fn local_op_action(op: &Op, units: &[Val], defs: Definitions<'_>) -> Option<LocalOp> {
    let (local_units, regions) = local_region_units(op)?;
    let local_units = dialects::expand_all_groups_to_units(&local_units, defs);
    if !local_units.iter().any(|unit| units.contains(unit)) {
        return Some(LocalOp::Delete);
    }
    // `else if (local_op->getNumRegions() == 1)`, the sizes agreeing and every local unit present.
    if regions == 1
        && local_units.len() == units.len()
        && local_units.iter().all(|unit| units.contains(unit))
    {
        return Some(LocalOp::Extract);
    }
    None
}

/// `getUnits()` — every region's list concatenated, as the op carries it — and `getNumRegions()`.
///
/// ⭐ BOTH ISLAND SPELLINGS: a local region whose body has reached this rung is a
/// [`dialects::UniformRegions`] and one that has not is a [`uniform::Op`], and the reference's
/// `dyn_cast` does not distinguish them.
fn local_region_units(op: &Op) -> Option<(Vec<Val>, usize)> {
    let flat = |regions: usize, units: Vec<Val>| Some((units, regions));
    match op {
        Op::UniformRegions(regions) => {
            let regions = regions.regions();
            flat(
                regions.len(),
                regions.iter().flat_map(|r| r.units.iter().copied()).collect(),
            )
        }
        Op::Uniform(
            uniform::Op::UniformizeRegions { regions, .. }
            | uniform::Op::EqualizePattern { regions },
        ) => flat(
            regions.len(),
            regions.iter().flat_map(|r| r.units.iter().copied()).collect(),
        ),
        _ => None,
    }
}

/// `dcc::uniform::utils::extractOpFromLocalRegion` (`dcc/src/Dialect/Uniform/Utils.cpp:1110-1147`)
/// with the caller's own erase (`Deuniform.cpp:208`) — the region's body MOVES to where the op stood.
///
/// ⭐ CLONE-THEN-ERASE IS A MOVE, so the reference's `else` branch —
/// `op.getResult(i).replaceAllUsesWith(new_op->getResult(i))` — is the identity and nothing is minted.
/// Only the `uniform_op.getResult(index)` rewires survive, and they are returned rather than applied.
fn extract_op_from_local_region(
    body: &mut Vec<Op>,
    at: usize,
    prog_arg: Val,
) -> (usize, Vec<(Val, Val)>) {
    let Some((arg, mut region, results)) = only_region(body.remove(at)) else {
        return (0, Vec::new());
    };
    // `block.getArgument(0).replaceAllUsesWith(prog_unit_arg)` (`:1119-1121`).
    dialects::replace_all_uses_with(&mut region, arg, prog_arg);
    // `uniform_results_val` — the yield's operands, positionally the op's own results (`:1123-1131`).
    let mut yielded = Vec::new();
    if let Some(Op::Uniform(uniform::Op::Yield { operands })) = region.last() {
        yielded = operands.clone();
        region.pop();
    }
    let mut rewires = Vec::new();
    for op in &region {
        for produced in dialects::results(op) {
            if let Some(index) = yielded.iter().position(|v| *v == produced)
                && let Some(result) = results.get(index)
            {
                rewires.push((*result, produced));
            }
        }
    }
    let moved = region.len();
    drop(body.splice(at..at, region));
    (moved, rewires)
}

/// The single region of a list, or `None` — the `getNumRegions() == 1` the caller has already asked.
///
/// ⭐ GENERIC BECAUSE THE ISLAND HAS TWO REGION TYPES, one per rung — see [`local_region_units`].
fn only<T>(mut regions: Vec<T>) -> Option<T> {
    (regions.len() == 1).then(|| regions.pop()).flatten()
}

/// The one region of a local-region op — its argument, its body at THIS rung, and the values the op
/// binds. ⭐ A LOWER-RUNG BODY COMES UP THROUGH [`dialects::raised`], so both spellings extract.
fn only_region(op: Op) -> Option<(Val, Vec<Op>, Vec<Val>)> {
    match op {
        Op::UniformRegions(dialects::UniformRegions::UniformizeRegions { regions, results }) => {
            let region = only(regions)?;
            // ⭐ ONLY THE VALUES — the op is about to be erased, so its register slots go with it.
            let results = results.into_iter().map(|result| result.val).collect();
            Some((region.arg, region.body, results))
        }
        Op::UniformRegions(dialects::UniformRegions::EqualizePattern { regions }) => {
            let region = only(regions)?;
            Some((region.arg, region.body, Vec::new()))
        }
        Op::Uniform(uniform::Op::UniformizeRegions { regions, results }) => {
            let region = only(regions)?;
            Some((
                region.arg,
                region.body.into_iter().map(dialects::raised).collect(),
                results,
            ))
        }
        Op::Uniform(uniform::Op::EqualizePattern { regions }) => {
            let region = only(regions)?;
            Some((
                region.arg,
                region.body.into_iter().map(dialects::raised).collect(),
                Vec::new(),
            ))
        }
        _ => None,
    }
}

/// Every `uniform.query_map` of the unit that can answer only one value, and that value — the second
/// walk's condition (`Deuniform.cpp:211-219`).
fn query_map_folds(
    body: &[Op],
    prog_arg: Val,
    units: &[Val],
    defs: Definitions<'_>,
) -> Vec<(Val, Val)> {
    let mut folds = Vec::new();
    for op in body {
        if let Op::Uniform(uniform::Op::QueryMap { result, map, key }) = op
            && let [answer] = dialects::uniform_mapping_values(*map, *key, defs)[..]
            && *key == prog_arg
            && dialects::uniform_mapping_keys(*key, defs).len() <= units.len()
        {
            folds.push((*result, answer));
        }
        for region in dialects::regions_ref(op) {
            folds.extend(query_map_folds(region, prog_arg, units, defs));
        }
    }
    folds
}

/// Every `uniform.def_immutable_mapping` of the unit nothing reads any more — `mapping_op.use_empty()`
/// (`Deuniform.cpp:223-226`).
fn unread_mappings(body: &[Op]) -> Vec<Val> {
    let mut mappings = Vec::new();
    collect_mappings(body, &mut mappings);
    mappings.retain(|result| dialects::use_count(*result, body) == 0);
    mappings
}

/// The handles every `uniform.def_immutable_mapping` in the tree binds.
fn collect_mappings(body: &[Op], into: &mut Vec<Val>) {
    for op in body {
        if let Op::Uniform(uniform::Op::DefImmutableMapping { result, .. }) = op {
            into.push(*result);
        }
        for region in dialects::regions_ref(op) {
            collect_mappings(region, into);
        }
    }
}

/// The two ops `duplicateAndUpdateEntriesOfQueryMap` builds, and what the old query map's readers are
/// re-pointed at.
pub(crate) struct DuplicatedQueryMap {
    /// The `uniform.def_immutable_mapping` and then the `uniform.query_map`, in insertion order.
    pub(crate) ops: [Op; 2],
    /// The value the new `uniform.query_map` binds — `new_query_map_op`'s result.
    pub(crate) result: Val,
}

/// Replaces: e298_duplicateAndUpdateEntriesOfQueryMap
///
/// A `uniform.query_map` rebuilt over a mapping carrying only the entries whose key survives in
/// `new_units`, reading the same key as before.
///
/// TRAP: the reference indexes `values[i]` by the KEY index while
/// `getListOfValueOpsFromUniformMapping` DROPS the keys the mapping has no entry for
/// (`Uniform.cpp:548-556`) — so a mapping with a hole reads past the end there. Here the pairing stops
/// at the shorter of the two lists.
pub(crate) fn duplicate_and_update_entries_of_query_map(
    map: Val,
    key: Val,
    new_units: &[Val],
    defs: Definitions<'_>,
    vals: &mut Values,
) -> DuplicatedQueryMap {
    let keys = dialects::uniform_mapping_keys(key, defs);
    let pairs = keys
        .into_iter()
        .zip(dialects::uniform_mapping_values(map, key, defs))
        .filter(|(mapped, _)| new_units.contains(mapped))
        .collect();
    let new_map = vals.mint();
    let result = vals.mint();
    DuplicatedQueryMap {
        ops: [
            Op::Uniform(uniform::Op::DefImmutableMapping {
                result: new_map,
                pairs,
            }),
            Op::Uniform(uniform::Op::QueryMap {
                result,
                map: new_map,
                key,
            }),
        ],
        result,
    }
}

/// ONE ENTRY OF A LOCAL-REGION OP'S `$list_sizes` — how many of the flat `$units` list belong to one
/// region (`Uniform.td:86-87`).
///
/// ⭐ A LENGTH AND NOT THE `i32` THE ATTRIBUTE HOLDS, because every read the reference makes of it is
/// a length: a loop bound over `$units` and a prefix-sum index into it (`Deuniform.cpp:257-278`,
/// `Uniform.cpp:184-192`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RegionUnitCount(pub(crate) usize);

/// Replaces: e299_expandAllGroupsToUnitsAndUpdateSizes
///
/// Every `dataflow.create_group` in a local-region op's flat `$units` replaced by its members, with
/// that region's `$list_sizes` entry raised to the member count.
///
/// TRAP: the reference's first loop builds `units_in_groups` and never reads it
/// (`Deuniform.cpp:250-257`) — dead there, so absent here.
pub(crate) fn expand_all_groups_to_units_and_update_sizes(
    units: &mut Vec<Val>,
    list_sizes: &mut Vec<RegionUnitCount>,
    defs: Definitions<'_>,
) {
    let mut new_units = Vec::new();
    let mut new_list_sizes = Vec::new();
    let mut rest = units.iter().copied();
    for size in list_sizes.iter() {
        let mut is_group = false;
        for unit in rest.by_ref().take(size.0) {
            match defs.of(unit) {
                Some(Op::Dataflow(dataflow::Op::CreateGroup { unit_ids, .. })) => {
                    is_group = true;
                    if size.0 != 1 {
                        todo!(
                            "expandAllGroupsToUnitsAndUpdateSizes: \
                             DT_CHECK(list_sizes[s].getInt() == 1) (Deuniform.cpp:263) — a group \
                             shares its region with {} further units",
                            size.0 - 1
                        );
                    }
                    new_units.extend(unit_ids.iter().copied());
                    new_list_sizes.push(RegionUnitCount(unit_ids.len()));
                }
                _ => new_units.push(unit),
            }
        }
        if !is_group {
            new_list_sizes.push(*size);
        }
    }
    *units = new_units;
    *list_sizes = new_list_sizes;
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

    use crate::islands::dataflow_ir::Units;
    use crate::islands::sentient::dialects::{LocalRegion, UniformRegions, sentient};
    use crate::units::{DfirUnit, Residency};

    /// A `dataflow.get_unit` binding `result`.
    fn get_unit(result: Val) -> Op {
        Op::Dataflow(dataflow::Op::GetUnit {
            result,
            residency: Residency::Global,
            unit: DfirUnit::L3lu,
            num_folds: None,
        })
    }

    /// `%result = sentient.scalar_copy %input` — the smallest op with one operand and one result.
    fn copy(input: Val, result: Val) -> Op {
        Op::Sentient(sentient::Op::ScalarCopy {
            input,
            result,
            reg: sentient::Reg {
                locale: sentient::RegType::Ebr,
                index: None,
            },
            element_size: None,
            program_header: false,
        })
    }

    /// A unit in the second set is found; one in no set is not.
    #[test]
    fn exists_in_collection_searches_every_set() {
        let collection = vec![vec![Val(1), Val(2)], vec![Val(3)]];

        assert!(exists_in_collection(Val(3), &collection));
        assert!(!exists_in_collection(Val(4), &collection));
    }

    /// All three phases at once on one `dataflow.program_unit` running on `%0`: the local region whose
    /// units are exactly `%0` is flattened and its yielded result rewired, the one on `%1` is dropped,
    /// and the query map on the unit's own argument folds to the mapping's one answer — which leaves
    /// the mapping unread and so erased.
    #[test]
    fn simplify_program_unit_op_flattens_drops_and_folds() {
        let preamble = vec![get_unit(Val(0)), get_unit(Val(1))];
        let mut unit = ProgramUnit::<crate::arch::Dd2> {
            on: Units::one(DfirUnit::L3lu, Val(0)),
            iter_arg: Some(Val(9)),
            precision: None,
            body: vec![
                // Exactly the program unit's units and one region: extracted, then erased.
                Op::UniformRegions(UniformRegions::UniformizeRegions {
                    regions: vec![LocalRegion {
                        arg: Val(20),
                        units: vec![Val(0)],
                        body: vec![
                            copy(Val(20), Val(11)),
                            Op::Uniform(uniform::Op::Yield {
                                operands: vec![Val(11)],
                            }),
                        ],
                    }],
                    results: vec![dialects::UniformResult::unallocated(Val(12))],
                }),
                // No unit in common: erased outright.
                Op::UniformRegions(UniformRegions::EqualizePattern {
                    regions: vec![LocalRegion {
                        arg: Val(21),
                        units: vec![Val(1)],
                        body: vec![Op::Uniform(uniform::Op::Yield {
                            operands: Vec::new(),
                        })],
                    }],
                }),
                Op::Uniform(uniform::Op::DefImmutableMapping {
                    result: Val(13),
                    pairs: vec![(Val(0), Val(14))],
                }),
                Op::Uniform(uniform::Op::QueryMap {
                    result: Val(15),
                    map: Val(13),
                    key: Val(9),
                }),
                copy(Val(12), Val(16)),
                copy(Val(15), Val(17)),
            ],
            arch: core::marker::PhantomData,
        };

        simplify_program_unit_op(&mut unit, &preamble);

        assert_eq!(
            unit.body,
            vec![
                // The region's own argument became the program unit's.
                copy(Val(9), Val(11)),
                // `%12`, the yielded result, is now the value the moved op produces.
                copy(Val(11), Val(16)),
                // The query map folded to the mapping's single answer.
                copy(Val(14), Val(17)),
            ]
        );
    }

    /// The vendor's own shape: a mapping over two units, rebuilt for a query whose key is one of them
    /// and whose surviving unit list holds only that one.
    #[test]
    fn duplicate_and_update_entries_of_query_map_keeps_only_surviving_keys() {
        let scope = vec![
            get_unit(Val(0)),
            get_unit(Val(1)),
            Op::Uniform(uniform::Op::DefImmutableMapping {
                result: Val(2),
                pairs: vec![(Val(0), Val(3)), (Val(1), Val(4))],
            }),
        ];
        let regions: [&[Op]; 1] = [&scope];
        let mut vals = Values::default();

        let duplicated = duplicate_and_update_entries_of_query_map(
            Val(2),
            Val(1),
            &[Val(1)],
            Definitions::from_innermost(&regions),
            &mut vals,
        );

        let [map, query] = duplicated.ops;
        assert_eq!(
            map,
            Op::Uniform(uniform::Op::DefImmutableMapping {
                result: Val(0),
                pairs: vec![(Val(1), Val(4))],
            })
        );
        assert_eq!(
            query,
            Op::Uniform(uniform::Op::QueryMap {
                result: Val(1),
                map: Val(0),
                key: Val(1),
            })
        );
        assert_eq!(duplicated.result, Val(1));
    }

    /// A two-region op whose first region is one `dataflow.create_group` of two members: that region
    /// grows to those two and its size follows, and the plain second region is untouched.
    #[test]
    fn expand_all_groups_to_units_and_update_sizes_raises_the_groups_region_size() {
        let scope = vec![
            get_unit(Val(0)),
            get_unit(Val(1)),
            get_unit(Val(2)),
            Op::Dataflow(dataflow::Op::CreateGroup {
                result: Val(3),
                unit_ids: vec![Val(0), Val(1)],
            }),
        ];
        let regions: [&[Op]; 1] = [&scope];
        let mut units = vec![Val(3), Val(2)];
        let mut list_sizes = vec![RegionUnitCount(1), RegionUnitCount(1)];

        expand_all_groups_to_units_and_update_sizes(
            &mut units,
            &mut list_sizes,
            Definitions::from_innermost(&regions),
        );

        assert_eq!(units, vec![Val(0), Val(1), Val(2)]);
        assert_eq!(list_sizes, vec![RegionUnitCount(2), RegionUnitCount(1)]);
    }
}

