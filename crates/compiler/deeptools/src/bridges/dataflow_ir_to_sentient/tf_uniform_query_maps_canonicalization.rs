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

//! `UniformQueryMapsCanonicalization.cpp` — 1 of bridge 2's 384 functions (dependency level(s) [2]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e261_runOnOperation` | 261/384 | 41 | `dcc/src/Transform/Dataflow/UniformQueryMapsCanonicalization.cpp:55` |

use super::tf_program_units_reduction::HighPreference;
use super::vc_vector_chain_helper::ops_are_equivalent;
use crate::islands::dataflow_ir::dialects::{
    Op as DfirOp, Val, dataflow, defining_op, regions, uniform, uses,
};

/// WHAT THE PASS DID TO ONE MODULE — the pass erases and replaces, so this is its whole output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalizedQueryMaps<'m> {
    /// `query_map_op.replaceAllUsesWith(val)` — each simplified query map's result and the map value
    /// every use of it becomes.
    pub substitutions: Vec<(Val, Val)>,
    /// The ops erased, IN THE REFERENCE'S OWN ORDER: each dead `uniform.query_map`, then the
    /// `uniform.def_immutable_mapping` that lost its last use, then that mapping's single-use values.
    pub erased: Vec<&'m DfirOp>,
}

/// EVERY `uniform.query_map` OF A MODULE, POST-ORDER — `module_op.walk(..)`, whose default
/// `WalkOrder::PostOrder` visits a region's ops before the op that holds them.
fn query_maps<'m>(scope: &'m [DfirOp], into: &mut Vec<&'m DfirOp>) {
    for op in scope {
        for region in regions(op) {
            query_maps(region, into);
        }
        if matches!(op, DfirOp::Uniform(uniform::Op::QueryMap { .. })) {
            into.push(op);
        }
    }
}

/// `getRegionOpAndIndex` (`dcc/src/Dialect/Uniform/Utils.cpp:343`) then `getUnitsOfRegion` (`:526`) —
/// the units of the INNERMOST enclosing region that maps one program onto units.
///
/// ⛔ NOT AN ANCHORED UNIT: `Dialect/Uniform/Utils.cpp` is outside this campaign's file list, and
/// [`run_on_operation`]'s simplification cannot be stated without it. ⚠️ The reference's walk stops at
/// a `uniform.equalize_pattern` too, which this island deliberately does not carry.
fn enclosing_region_units(query_map: Val, scope: &[DfirOp]) -> Option<Vec<Val>> {
    for op in scope {
        for (index, region) in regions(op).into_iter().enumerate() {
            if defining_op(query_map, region).is_none() {
                continue;
            }
            // `node = node->getParentOp()` walks OUTWARD, so a mapping op nested deeper answers first.
            if let Some(inner) = enclosing_region_units(query_map, region) {
                return Some(inner);
            }
            match op {
                DfirOp::Uniform(uniform::Op::UniformizeRegions { regions, .. }) => {
                    return regions.get(index).map(|region| region.units.clone());
                }
                // `if (auto prog_unit_op = dyn_cast<ProgramUnitOp>(region_op))` — its operands whole,
                // not sliced by `list_sizes`.
                DfirOp::Dataflow(dataflow::Op::ProgramUnit { units, .. }) => {
                    return Some(units.clone());
                }
                _ => {}
            }
        }
    }
    None
}

/// `simplifyQueryMapWithSameTarget` (`dcc/src/Dialect/Uniform/Utils.cpp:1263`) — the value every use
/// of one query map becomes, or `None` where the mapping still says different things.
///
/// ⛔ NOT AN ANCHORED UNIT, for [`enclosing_region_units`]' reason: it is the whole of what entry 261
/// does to a query map, and the pass is nothing without it. ⚠️ A map value defined by a region
/// ARGUMENT is a null `getDefiningOp()` the reference dereferences (`:1276-1277`); a mapping that
/// cannot be compared says different things, which leaves the IR exactly as it stands.
fn simplify_query_map_with_same_target(result: Val, map: Val, scope: &[DfirOp]) -> Option<Val> {
    // `DT_CHECK(map_op)` — the handle is a `uniform.def_immutable_mapping` or there is nothing to read.
    let DfirOp::Uniform(uniform::Op::DefImmutableMapping { pairs, .. }) = defining_op(map, scope)?
    else {
        return None;
    };

    let (_, val0) = *pairs.first()?;
    let val0_def = defining_op(val0, scope);
    let has_different_target = pairs.iter().any(|(_, val)| {
        match (defining_op(*val, scope), val0_def) {
            // `if (val.getDefiningOp() == val0.getDefiningOp() && val != val0)` — two results of one
            // op are two different targets.
            (Some(def), Some(def0)) if core::ptr::eq(def, def0) => *val != val0,
            // `if (!oe.operationsAreEquivalent(*val.getDefiningOp(), *val0.getDefiningOp()))`.
            (Some(def), Some(def0)) => !ops_are_equivalent(def, def0, scope, HighPreference::None),
            _ => true,
        }
    });

    // `if (!has_different_target) { query_map_op.replaceAllUsesWith(val0); return; }`
    if !has_different_target {
        return Some(val0);
    }

    // `if (units.size() == 1) { if (map_op.getValue(units[0]).has_value()) .. }` — one unit in the
    // region leaves one answer, whatever the rest of the map says.
    let units = enclosing_region_units(result, scope)?;
    let [unit] = units.as_slice() else {
        return None;
    };
    pairs
        .iter()
        .find(|(key, _)| key == unit)
        .map(|(_, value)| *value)
}

/// Replaces: e261_runOnOperation
///
/// **261/384** `UniformQueryMapsCanonicalizationPass::runOnOperation` — a query map whose mapping
/// says one thing becomes that thing, and the mapping goes with it when nothing else reads it.
///
/// ⛔ `hasOneUse()` IS ASKED OF THE **REWRITTEN** IR (`:79`): `replaceAllUsesWith` has already moved
/// every use of the simplified query maps onto their targets, so [`CanonicalizedQueryMaps::substitutions`] must be
/// counted in before a map value can be called dead. ⚠️ `EnableDeadMapVarDeletion` (default `true`)
/// and `DisableThisPass` are `cl::opt`s, not questions this crate asks at run time.
#[must_use]
pub fn run_on_operation(module: &[DfirOp]) -> CanonicalizedQueryMaps<'_> {
    let mut walked = Vec::new();
    query_maps(module, &mut walked);

    let mut substitutions = Vec::new();
    let mut query_maps_to_be_deleted = Vec::new();
    for op in walked {
        let DfirOp::Uniform(uniform::Op::QueryMap { result, map, .. }) = op else {
            continue;
        };
        let target = simplify_query_map_with_same_target(*result, *map, module);
        if let Some(target) = target {
            substitutions.push((*result, target));
        }
        // `if (query_map_op.getResult().getUses().empty())`, asked AFTER the replacement — so a
        // simplified query map is always one of these.
        if target.is_some() || uses(*result, module).is_empty() {
            query_maps_to_be_deleted.push(op);
        }
    }

    // `v.hasOneUse()` on the rewritten IR: the uses the module still shows, plus the ones
    // `replaceAllUsesWith` moved onto this value.
    let use_count = |val: Val| {
        uses(val, module).len()
            + substitutions
                .iter()
                .filter(|(_, target)| *target == val)
                .map(|(result, _)| uses(*result, module).len())
                .sum::<usize>()
    };

    // `for (auto it = ..rbegin(); it != ..rend(); ++it)`.
    let mut erased: Vec<&DfirOp> = Vec::new();
    for op in query_maps_to_be_deleted.iter().rev() {
        let DfirOp::Uniform(uniform::Op::QueryMap { map, .. }) = op else {
            continue;
        };
        // `it->erase();`
        erased.push(op);

        // `it->getMap().getDefiningOp<DefImmutableMappingOp>()`.
        let Some(def_map_op) = defining_op(*map, module) else {
            continue;
        };
        let DfirOp::Uniform(uniform::Op::DefImmutableMapping {
            result: handle,
            pairs,
            ..
        }) = def_map_op
        else {
            continue;
        };

        // `if (!def_map_op.getResult().getUses().empty()) continue;` — the query maps erased above no
        // longer count, which is what makes a shared mapping outlive all but its last reader.
        if uses(*handle, module)
            .into_iter()
            .any(|user| !erased.iter().any(|done| core::ptr::eq(*done, user)))
        {
            continue;
        }

        // `if (op && v.hasOneUse()) dead_ops.insert(op);` — a value the map alone reads dies with it.
        // ⚠️ THE REFERENCE'S `SmallPtrSet` HAS NO ORDER; ours is the mapping's own pair order.
        let dead_ops: Vec<&DfirOp> = pairs
            .iter()
            .filter(|(_, value)| use_count(*value) == 1)
            .filter_map(|(_, value)| defining_op(*value, module))
            .collect();

        // `def_map_op.erase();` then `for (Operation *op : dead_ops) op->erase();`.
        erased.push(def_map_op);
        erased.extend(dead_ops);
    }

    CanonicalizedQueryMaps {
        substitutions,
        erased,
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::islands::dataflow_ir::Values;
    use crate::islands::dataflow_ir::dialects::uniform::LocalRegion;
    use crate::islands::dataflow_ir::dialects::uniform::MappedTy;
    use crate::islands::dataflow_ir::dialects::{arith, results};
    use crate::islands::dataflow_ir::ty::ScalarTy;

    /// A MAPPING WHOSE TWO UNITS NAME THE SAME CONSTANT — the query map, the mapping and the ONE map
    /// value that nothing else reads all go, while `val0` stays because the query map's use moved onto
    /// it.
    #[test]
    fn a_mapping_that_says_one_thing_takes_itself_and_its_query_map_out() {
        let mut vals = Values::default();
        let k0 = vals.mint();
        let k1 = vals.mint();
        let v0 = vals.mint();
        let v1 = vals.mint();
        let handle = vals.mint();
        let arg = vals.mint();
        let queried = vals.mint();
        let sum = vals.mint();

        let constant =
            |result: Val, value: i64| DfirOp::Arith(arith::Op::Constant { result, value });
        let module = vec![
            constant(k0, 0),
            constant(k1, 1),
            constant(v0, 4),
            constant(v1, 4),
            DfirOp::Uniform(uniform::Op::DefImmutableMapping {
                result: handle,
                pairs: vec![(k0, v0), (k1, v1)],
                values_ty: MappedTy::Index,
            }),
            DfirOp::Uniform(uniform::Op::UniformizeRegions {
                regions: vec![LocalRegion {
                    arg,
                    units: vec![k0, k1],
                    body: vec![
                        DfirOp::Uniform(uniform::Op::QueryMap {
                            result: queried,
                            map: handle,
                            key: arg,
                            ty: MappedTy::Index,
                        }),
                        DfirOp::Arith(arith::Op::AddI(arith::IntBinary {
                            result: sum,
                            lhs: queried,
                            rhs: queried,
                            ty: ScalarTy::Index,
                        })),
                        DfirOp::Uniform(uniform::Op::Yield { operands: vec![] }),
                    ],
                }],
                results: vec![],
            }),
        ];

        let done = run_on_operation(&module);
        assert_eq!(vec![(queried, v0)], done.substitutions);
        // ⭐ `%v0` IS NOT DEAD: the two uses of `%queried` are now its own, which is exactly the
        // question `hasOneUse()` asks of the rewritten IR.
        assert_eq!(
            vec![vec![queried], vec![handle], vec![v1]],
            done.erased
                .iter()
                .map(|op| results(op))
                .collect::<Vec<Vec<Val>>>()
        );
    }
}
