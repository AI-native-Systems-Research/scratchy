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

//! `OldRegisterInitialization.cpp` — 7 of the campaign's 656 units (dependency level(s) [0, 1, 2, 3, 5]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e111_isWithinGlobalRegion` | 111 | 0 | 10 | `dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:1075` |
//! | `e112_isSameOpType` | 112 | 0 | 18 | `dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:1087` |
//! | `e333_constructValues` | 333 | 1 | 34 | `dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:1107` |
//! | `e452_postProcessing` | 452 | 2 | 27 | `dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:1047` |
//! | `e517_getSource` | 517 | 3 | 38 | `dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:1144` |
//! | `e518_replace` | 518 | 3 | 17 | `dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:1184` |
//! | `e606_run` | 606 | 5 | 224 | `dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:811` |

// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET, so everything below is reachable only from this
// file's own tests until `e606_run` lands and something calls it.
// ⭐ REMOVE THIS WITH `e606_run`: at that point an unused item here is a real defect again.
#![allow(dead_code)]

use crate::islands::dataflow_ir::dialects as lower;
use crate::islands::sentient::dialects::{
    self, Definitions, Op, Val, dataflow, sentient, symbol, uniform,
};
use crate::transform::sentient::analyses::UniformGroups;
use crate::transform::sentient::utils::units_and_their_values::UnitsAndTheirValues;

/// WHERE THE OP BEHIND A CANDIDATE SITS — what `getParentOfType<uniform::UniformizeRegionsOp,
/// uniform::EqualizePatternOp>` answers about it (`:1081-1083`).
///
/// ⛔ `getParentOfType` IS STRICT IN THIS TREE: it starts at `op.getParentOp()`, so a
/// `uniform.uniformize_regions` is not inside ITSELF and neither is the region argument whose owner's
/// parent op it is — a [`uniform::LocalRegion::arg`] is GLOBAL.
/// ⛔ `uniform.equalize_pattern` HAS NO ISLAND OP, the same recorded gap entry 020 carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ancestry {
    /// Nothing in this subtree defines the value or binds it as a region argument.
    Unbound,
    /// No `uniform.uniformize_regions` strictly encloses the op behind it.
    Global,
    /// One does.
    UnderUniformRegion,
}

/// Whether `op` binds `val` as a REGION ARGUMENT, the reference's `dyn_cast<BlockArgument>` case —
/// and every one of them answers for the binding op's own scope, because that op IS
/// `arg.getOwner()->getParentOp()`.
fn binds_as_region_arg(op: &Op, val: Val) -> bool {
    match op {
        Op::Sentient(sentient::Op::For { iv, carried, .. }) => {
            *iv == val || carried.iter().any(|position| position.arg == val)
        }
        Op::AffineFor(loop_op) => {
            loop_op.iv == val || loop_op.carried.iter().any(|position| position.arg == val)
        }
        Op::Uniform(uniform::Op::UniformizeRegions { regions, .. }) => {
            regions.iter().any(|region| region.arg == val)
        }
        // THE SAME `$arg` LIST, one rung up.
        Op::UniformRegions(regions) => regions.regions().iter().any(|region| region.arg == val),
        _ => false,
    }
}

/// The walk behind [`is_within_global_region`], carrying whether `scope`'s ops are already inside a
/// `uniform.uniformize_regions`.
fn ancestry(val: Val, scope: &[Op], under_uniform: bool) -> Ancestry {
    let bound_here = if under_uniform {
        Ancestry::UnderUniformRegion
    } else {
        Ancestry::Global
    };
    for op in scope {
        if dialects::results(op).contains(&val) || binds_as_region_arg(op, val) {
            return bound_here;
        }
        match op {
            Op::Sentient(inner) => {
                for region in sentient::regions(inner) {
                    match ancestry(val, region, under_uniform) {
                        Ancestry::Unbound => {}
                        found => return found,
                    }
                }
            }
            Op::AffineFor(loop_op) => match ancestry(val, &loop_op.body, under_uniform) {
                Ancestry::Unbound => {}
                found => return found,
            },
            // ⭐ THE OTHER ARM THAT CAN ANSWER `false`, and its regions ARE this rung's — so a
            // `sentient.*` value sunk into a local region is answered here, not guessed at.
            Op::UniformRegions(regions) => {
                for region in regions.regions() {
                    match ancestry(val, &region.body, true) {
                        Ancestry::Unbound => {}
                        found => return found,
                    }
                }
            }
            // ⭐ THE ONE ARM THAT CAN ANSWER `false` A RUNG LOWER: a local region's body holds
            // [`lower::Op`], so the candidates reachable inside one are the SHARED ops —
            // `dataflow.get_unit`, `symbol.create_symbol`, `uniform.query_map` — which is exactly the
            // set entry 112 pairs.
            Op::Uniform(uniform::Op::UniformizeRegions { regions, .. }) => {
                if regions
                    .iter()
                    .any(|region| lower::defining_op(val, &region.body).is_some())
                {
                    return Ancestry::UnderUniformRegion;
                }
            }
            other => {
                if let Some(op) = dialects::lowered(other)
                    && lower::defining_op(val, core::slice::from_ref(&op)).is_some()
                {
                    todo!(
                        "isWithinGlobalRegion reached a definition inside a lower-rung region: {op:?}"
                    );
                }
            }
        }
    }
    Ancestry::Unbound
}

/// Replaces: e111_isWithinGlobalRegion
///
/// Whether a candidate comes from outside every local region — the test the promoter drains its
/// candidate stack with, one global value at a time (`:844-853`).
#[must_use]
pub fn is_within_global_region(val: Val, scope: &[Op]) -> bool {
    match ancestry(val, scope, false) {
        Ancestry::Global => true,
        Ancestry::UnderUniformRegion => false,
        // `DT_CHECK(op)` (`:1080`) — the reference aborts on a value with nothing behind it. The one
        // shape this island cannot answer for is a region argument bound INSIDE a local region's
        // lower-rung body, which no `getRegionIterArgs()` candidate of this rung is.
        Ancestry::Unbound => todo!(
            "isWithinGlobalRegion: DT_CHECK(op) — nothing in the module binds {val:?} (OldRegisterInitialization.cpp:1080)"
        ),
    }
}

/// THE OPS A SHARED REGISTER INIT MAY BE BUILT FROM — the four `isa<>` pairs of `:1092-1101`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SourceOpType {
    /// `sentient::ConstantOp`, this island's `sentient.scalar_constant`.
    ScalarConstant,
    /// `mlir::symbol::CreateSymbolOp`.
    CreateSymbol,
    /// `dataflow::GetUnitOp`.
    GetUnit,
    /// `dataflow::CreateMulticastGroupOp`.
    CreateMulticastGroup,
}

/// Which of them defines `val`, or nothing for any other op AND for a value with no defining op.
///
/// ⛔ DELIBERATE DIVERGENCE ON A NULL DEFINING OP: the reference's four tests are BARE `isa<>` on an
/// `Operation *` (`:1092-1101`), which ASSERTS on null rather than answering false — `dcc` spells the
/// null-tolerant form `dyn_cast_or_null` wherever it means it (`:220`, `:222`, `:227`). Both callers
/// pass values `getSource` built (`:864-865`, `:879-880`), so a region argument does not reach it;
/// answering `None` keeps this total instead of reproducing an assert.
fn source_op_type(val: Val, scope: &[Op]) -> Option<SourceOpType> {
    match dialects::defining_op(val, scope)? {
        Op::Sentient(sentient::Op::ScalarConstant { .. }) => Some(SourceOpType::ScalarConstant),
        Op::Symbol(symbol::Op::CreateSymbol { .. }) => Some(SourceOpType::CreateSymbol),
        Op::Dataflow(dataflow::Op::GetUnit { .. }) => Some(SourceOpType::GetUnit),
        Op::Dataflow(dataflow::Op::CreateMulticastGroup { .. }) => {
            Some(SourceOpType::CreateMulticastGroup)
        }
        _ => None,
    }
}

/// Replaces: e112_isSameOpType
///
/// Whether two sources are the same kind of op — and `true` for a pair where either side is null,
/// which is the reference's own answer for one.
#[must_use]
pub fn is_same_op_type(val1: Option<Val>, val2: Option<Val>, scope: &[Op]) -> bool {
    match (val1, val2) {
        (Some(val1), Some(val2)) => {
            match (source_op_type(val1, scope), source_op_type(val2, scope)) {
                (Some(type1), Some(type2)) => type1 == type2,
                _ => false,
            }
        }
        _ => true,
    }
}

/// `def_map.getNonNullValuesFromKeys({key})` for ONE key — the entry a
/// `uniform.def_immutable_mapping` holds for it, or nothing (`dataflow-scheduler/.../lib/Dialect/
/// Uniform/Uniform.cpp:548-556`).
fn mapped_value(map: Val, key: Val, defs: Definitions<'_>) -> Option<Val> {
    let Some(Op::Uniform(uniform::Op::DefImmutableMapping { pairs, .. })) = defs.of(map) else {
        return None;
    };
    pairs
        .iter()
        .find(|(mapped, _)| *mapped == key)
        .map(|(_, value)| *value)
}

/// THE UNITS ONE `dataflow.get_unit` STANDS FOR — `get_unit_op->getResults()`.
///
/// ⛔ ONE, AND THAT IS A RECORDED ISLAND GAP, NOT A CHOICE. `dataflow.get_unit` binds
/// `Variadic<Index>:$units` in the reference (`Dataflow.td:56-58`) while this island's
/// [`dataflow::Op::GetUnit`] binds a single [`Val`] — the same gap
/// `bridges/dataflow_ir_to_sentient/tf_unit_filtering.rs:1211-1274` records, where the rebuild's only
/// observable is `num_folds`. So `nfolds` is 1 here and the fold loop runs once.
fn folds_of(core: Val) -> Vec<Val> {
    vec![core]
}

/// Replaces: e333_constructValues
///
/// Pairs every unit a candidate must be shared across with the value that unit needs: one entry per
/// fold of `core`'s `dataflow.get_unit`, plus one for each non-leader unit `core` speaks for
/// (`OldRegisterInitialization.cpp:1107-1141`).
///
/// ⛔ TRAP: A UNIFORMIZED CANDIDATE IS PER-UNIT AND EVERY OTHER IS SHARED. Behind a
/// `uniform.query_map` each key gets its OWN mapped value; behind anything else every key gets the
/// SAME `val`.
/// ⛔ TRAP: A FOLLOWER RESOLVING TO `core`'S OWN `get_unit` IS SKIPPED — it is already one of the
/// folds above, and adding it twice would make `areAllValuesEqual` (`e253`) count one unit twice.
/// ⚠️ `DT_CHECK(tmp_values.size() == tmp_keys.size())` BECOMES A HOLE: `getNonNullValuesFromKeys` drops
/// a key the map has no entry for, so a key with no entry is added with `None` here — identical
/// whenever the check holds, and `None` is what [`UnitsAndTheirValues`] already spells a null value.
pub fn construct_values<U: UniformGroups>(
    val: Val,
    core: Val,
    results: &mut UnitsAndTheirValues,
    defs: Definitions<'_>,
    uga: &U,
) {
    let mut keys = folds_of(core);
    // `uga_.isGroupLeader(core)` then `getGroupMembersLedBy(core)` — the units this one answers for.
    if uga.is_group_leader(core) {
        for follower in uga.group_members_led_by(core) {
            if follower != core {
                keys.push(follower);
            }
        }
    }
    match defs.of(val) {
        Some(Op::Uniform(uniform::Op::QueryMap { map, .. })) => {
            let map = *map;
            for key in keys {
                results.add(key, mapped_value(map, key, defs));
            }
        }
        _ => {
            for key in keys {
                results.add(key, Some(val));
            }
        }
    }
}

// crustify:todo: e452_postProcessing
//   authority : dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:1047  (27 body lines, level 2)
//   original  : void RegisterInitCandidatePromoter::postProcessing()
//   calls     : e109_eraseDeletedOps, e326_replaceVirtualAssignTarget, e332_moveSSAToInit

// crustify:todo: e517_getSource
//   authority : dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:1144  (38 body lines, level 3)
//   original  : void RegisterInitCandidatePromoter::getSource( mlir::Value core, mlir::Value val, OpBuilder &const_builder, dcc::utils::UnitsAndTheirValues &results)
//   calls     : e252_size, e333_constructValues, e396_replaceValue, e422_insert

// crustify:todo: e518_replace
//   authority : dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:1184  (17 body lines, level 3)
//   original  : void RegisterInitCandidatePromoter::replace(mlir::Value val, mlir::Value copy_op_result)
//   calls     : e422_insert

// crustify:todo: e606_run
//   authority : dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:811  (224 body lines, level 5)
//   original  : void RegisterInitCandidatePromoter::run()
//   calls     : e063_getMaxRegNum, e108_removeInitAttrFromOps, e109_eraseDeletedOps, e110_hasSameAttr, e111_isWithinGlobalRegion, e112_isSameOpType, e249_normalizeNullValues, e251_add, e252_size, e253_areAllValuesEqual, e332_moveSSAToInit, e422_insert, e452_postProcessing, e517_getSource …

#[cfg(test)]
mod unit_tests {
    use super::{construct_values, is_same_op_type, is_within_global_region};
    use crate::islands::dataflow_ir::dialects as lower;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::{
        Definitions, Op, Val, dataflow, sentient, symbol, uniform,
    };
    use crate::transform::sentient::analyses::UniformGroups;
    use crate::transform::sentient::utils::units_and_their_values::UnitsAndTheirValues;
    use crate::units::{DfirUnit, Residency};

    /// The out-of-scope group analysis, answering for ONE leader and its followers.
    struct Groups {
        leader: Val,
        followers: Vec<Val>,
    }

    impl UniformGroups for Groups {
        fn group_leaders(&self) -> Vec<Val> {
            vec![self.leader]
        }

        fn is_group_leader(&self, unit: Val) -> bool {
            unit == self.leader
        }

        fn group_members_led_by(&self, leader: Val) -> Vec<Val> {
            if leader == self.leader {
                self.followers.clone()
            } else {
                Vec::new()
            }
        }
    }

    /// `%r = dataflow.get_unit`, at whichever rung the region holding it is.
    fn get_unit(result: u32) -> dataflow::Op {
        dataflow::Op::GetUnit {
            result: Val(result),
            residency: Residency::Global,
            unit: DfirUnit::L3lu,
            num_folds: None,
        }
    }

    /// `%r = dataflow.create_multicast_group(%1 -> ())`.
    fn multicast_group(result: u32) -> dataflow::Op {
        dataflow::Op::CreateMulticastGroup {
            result: Val(result),
            producer: Val(1),
            consumers: Vec::new(),
            num_consumers: dataflow::ConsumerCount(1),
            group_id: dataflow::MulticastGroupId(0),
            count: dataflow::OutstandingRequests(0),
            // ⭐ ABSENT, WHICH IS THE VENDOR'S DEFAULT — see
            // [`dataflow::Op::CreateMulticastGroup::init_packet_opt_en`].
            init_packet_opt_en: false,
        }
    }

    /// `%r = sentient.scalar_constant`.
    fn scalar_constant(result: u32) -> Op {
        Op::Sentient(sentient::Op::ScalarConstant {
            value: 0,
            result: Val(result),
            reg_locale: sentient::RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    /// e111 — a global unit, one bound inside a local region, and the region argument the strict
    /// `getParentOfType` leaves global.
    #[test]
    fn e111_answers_no_only_for_a_value_bound_inside_a_local_region() {
        let scope = vec![
            Op::Dataflow(get_unit(1)),
            Op::Uniform(uniform::Op::UniformizeRegions {
                regions: vec![uniform::LocalRegion {
                    arg: Val(2),
                    units: vec![Val(1)],
                    body: vec![lower::Op::Dataflow(get_unit(3))],
                }],
                results: Vec::new(),
            }),
            Op::Sentient(sentient::Op::For {
                iv: Val(4),
                bound: Val(1),
                bound_reg: None,
                carried: Vec::new(),
                dbg_name: None,
                body: vec![scalar_constant(5)],
            }),
        ];
        assert!(is_within_global_region(Val(1), &scope));
        assert!(!is_within_global_region(Val(3), &scope));
        assert!(is_within_global_region(Val(2), &scope));
        // A loop's own induction variable and a value inside its body are both global — the loop is
        // not a local region.
        assert!(is_within_global_region(Val(4), &scope));
        assert!(is_within_global_region(Val(5), &scope));
    }

    /// e333 — a plain candidate is shared with every unit, and a uniformized one is looked up PER
    /// unit, with the follower that is `core` itself skipped.
    #[test]
    fn e333_shares_one_value_but_looks_a_uniformized_one_up_per_unit() {
        let scope = vec![
            Op::Dataflow(get_unit(1)),
            Op::Dataflow(get_unit(2)),
            scalar_constant(3),
            scalar_constant(4),
            Op::Uniform(uniform::Op::DefImmutableMapping {
                result: Val(5),
                pairs: vec![(Val(1), Val(3)), (Val(2), Val(4))],
            }),
            Op::Uniform(uniform::Op::QueryMap {
                result: Val(6),
                map: Val(5),
                key: Val(1),
            }),
        ];
        let module = [scope.as_slice()];
        let defs = Definitions::from_innermost(&module);
        let groups = Groups {
            leader: Val(1),
            // ⭐ `core` ITSELF AMONG ITS OWN FOLLOWERS: skipped, because the fold loop added it.
            followers: vec![Val(1), Val(2)],
        };

        let mut shared = UnitsAndTheirValues::default();
        construct_values(Val(3), Val(1), &mut shared, defs, &groups);
        assert_eq!(
            shared.pairs,
            vec![(Val(1), Some(Val(3))), (Val(2), Some(Val(3)))]
        );

        let mut per_unit = UnitsAndTheirValues::default();
        construct_values(Val(6), Val(1), &mut per_unit, defs, &groups);
        assert_eq!(
            per_unit.pairs,
            vec![(Val(1), Some(Val(3))), (Val(2), Some(Val(4)))]
        );

        // ⚠️ A KEY THE MAP HAS NO ENTRY FOR IS THE DROPPED `getNonNullValuesFromKeys` HOLE.
        let mut hole = UnitsAndTheirValues::default();
        let unmapped = Groups {
            leader: Val(1),
            followers: vec![Val(9)],
        };
        construct_values(Val(6), Val(1), &mut hole, defs, &unmapped);
        assert_eq!(hole.pairs, vec![(Val(1), Some(Val(3))), (Val(9), None)]);
    }

    /// e112 — a matching pair, a mismatched one, the null side, and the region argument whose
    /// `isa<T>(nullptr)` is false for all four ops.
    #[test]
    fn e112_pairs_only_the_named_ops_and_says_yes_to_a_null_side() {
        let scope = vec![
            scalar_constant(1),
            scalar_constant(2),
            Op::Dataflow(get_unit(3)),
            Op::Symbol(symbol::Op::CreateSymbol {
                result: Val(4),
                symbol_id: 0,
                max_value: None,
            }),
            Op::Sentient(sentient::Op::For {
                iv: Val(5),
                bound: Val(3),
                bound_reg: None,
                carried: Vec::new(),
                dbg_name: None,
                body: Vec::new(),
            }),
            Op::Dataflow(multicast_group(6)),
            Op::Dataflow(multicast_group(7)),
        ];
        assert!(is_same_op_type(Some(Val(1)), Some(Val(2)), &scope));
        assert!(is_same_op_type(Some(Val(6)), Some(Val(7)), &scope));
        assert!(!is_same_op_type(Some(Val(1)), Some(Val(3)), &scope));
        assert!(!is_same_op_type(Some(Val(6)), Some(Val(4)), &scope));
        assert!(!is_same_op_type(Some(Val(3)), Some(Val(4)), &scope));
        assert!(is_same_op_type(Some(Val(1)), None, &scope));
        assert!(is_same_op_type(None, None, &scope));
        // ⛔ THE DIVERGENCE, PINNED: a value with no defining op is none of the four rather than
        // the reference's assert. See [`is_same_op_type`].
        assert!(!is_same_op_type(Some(Val(5)), Some(Val(5)), &scope));
    }
}
