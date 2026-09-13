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

//! `SymbolToSentient.cpp` — 3 of bridge 2's 384 functions (dependency level(s) [2, 3, 4]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e226_createIfOpFromMapping` | 226/384 | 61 | `dcc/src/Conversion/SymbolToSentient/SymbolToSentient.cpp:123` |
//! | `e275_LowerSymbolQueryMap` | 275/384 | 68 | `dcc/src/Conversion/SymbolToSentient/SymbolToSentient.cpp:40` |
//! | `e303_runOnOperation` | 303/384 | 15 | `dcc/src/Conversion/SymbolToSentient/SymbolToSentient.cpp:23` |

use crate::arch::Arch;
use crate::islands::dataflow_ir::dialects::{
    Op as DfirOp, Val, affine, defining_op, regions, scf, symbol, uses,
};
use crate::islands::dataflow_ir::ty::GenericComp;
use crate::islands::dataflow_ir::{Program, Values};
use crate::islands::sentient::dialects::Op as SenOp;
use crate::islands::sentient::dialects::sentient as sen;

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 226/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// THE MAPPING A `symbol.query_map` READS, SHAPED SO THE `DT_CHECK_MSG` HAS NO INPUT.
///
/// ⛔⛔ AT LEAST TWO PAIRS, AND THE LAST ONE'S KEY IS NEVER COMPARED. `DT_CHECK_MSG(key_list.size()
/// >= 2, "Expect query_map with at least 2 mappings")` (`SymbolToSentient.cpp:134`) is discharged by
/// [`Self::first`] plus [`Self::last_value`] both existing; and `if (rhs_val == last_map_key) break;`
/// (`:153`) means the final pair contributes only its VALUE, as the innermost `else`. Holding a flat
/// `Vec` of pairs would make both of those facts run-time questions.
#[derive(Debug, Clone, Copy)]
pub struct QueryMapping<'a> {
    /// `query_map.getKey()` — the LHS of every comparison in the chain.
    pub key: Val,
    /// The first `(key, value)` pair — the outermost conditional.
    pub first: (Val, Val),
    /// The pairs between the first and the last, in order.
    pub middle: &'a [(Val, Val)],
    /// The last pair's value. ⛔ ITS KEY IS ONLY THE LOOP'S BREAK SENTINEL and is never compared.
    pub last_value: Val,
}

/// Replaces: e226_createIfOpFromMapping
///
/// **226/384** `SymbolToSentientLoweringPass::createIfOpFromMapping` — `dcc/src/Conversion/SymbolToSentient/SymbolToSentient.cpp:123` (61L).
///
/// The mapping as a right-nested chain of `sentient.if (key == k_i)`, each arm yielding its value
/// `num_results` times, the innermost `else` yielding [`QueryMapping::last_value`], and every
/// intermediate `else` yielding the nested conditional's own results (`:167-168`). Returns the
/// OUTERMOST conditional — the reference's `new_if_op`, captured on the first pair only.
#[must_use]
pub fn create_if_op_from_mapping(
    mapping: QueryMapping<'_>,
    num_results: usize,
    values: &mut Values,
) -> sen::Op {
    nest(
        mapping.key,
        mapping.first,
        mapping.middle,
        mapping.last_value,
        num_results,
        values,
    )
    .0
}

/// One conditional of [`create_if_op_from_mapping`]'s chain and the results it binds.
///
/// ⭐ THE RESULTS ARE MINTED BEFORE THE RECURSION, so the outermost conditional binds the lowest
/// values — the order the reference's `else_builder` creates them in.
fn nest(
    key: Val,
    head: (Val, Val),
    rest: &[(Val, Val)],
    last_value: Val,
    num_results: usize,
    values: &mut Values,
) -> (sen::Op, Vec<Val>) {
    let results: Vec<Val> = (0..num_results).map(|_| values.mint()).collect();
    let else_body = match rest.split_first() {
        // *"The innermost else-branch will yield num_result of the last value in the list."*
        None => vec![SenOp::Sentient(sen::Op::Yield {
            results: vec![last_value; num_results],
        })],
        Some((&next_head, next_rest)) => {
            let (inner, inner_results) =
                nest(key, next_head, next_rest, last_value, num_results, values);
            vec![
                SenOp::Sentient(inner),
                SenOp::Sentient(sen::Op::Yield {
                    results: inner_results,
                }),
            ]
        }
    };
    let op = sen::Op::If {
        // `CmpIPredicateAttr::get(.., CmpIPredicate::eq)`.
        predicate: sen::CmpPredicate::Eq,
        lhs: key,
        rhs: head.0,
        // `locale_attr` — `num_results` copies of `SentientRegType::unknown`.
        yielded: results
            .iter()
            .map(|result| sen::Yielded {
                result: *result,
                reg: sen::Reg {
                    locale: sen::RegType::Unknown,
                    index: None,
                },
                element_size: None,
            })
            .collect(),
        dbg_name: None,
        then_body: vec![SenOp::Sentient(sen::Op::Yield {
            results: vec![head.1; num_results],
        })],
        // ⭐ `/*has else branch*/ true` — every conditional in this chain has one.
        else_body,
    };
    (op, results)
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 275/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// THE MAPPING A `symbol.query_map` READS, IN THE TWO SHAPES `LowerSymbolQueryMap` DISTINGUISHES.
///
/// ⛔ THE ONE-PAIR CASE IS A DIFFERENT ANSWER, NOT A DEGENERATE CHAIN
/// (`SymbolToSentient.cpp:50-53`), and [`QueryMapping`] cannot hold one pair — so the split is the
/// input type rather than a size test, and `createIfOpFromMapping`'s
/// `DT_CHECK_MSG(key_list.size() >= 2)` has no reachable input.
#[derive(Debug, Clone, Copy)]
pub enum SymbolMapping<'a> {
    /// `immutable_mapping.getKeys().size() == 1` — its only value.
    Single(Val),
    /// Two or more pairs.
    Chain(QueryMapping<'a>),
}

/// WHAT ONE USE OF THE `symbol.query_map` RESULT IS — the classification the replacement loop
/// repeats (`SymbolToSentient.cpp:67-73`, `:98-103`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryMapUse {
    /// The `bound` operand of a `sentient.for` —
    /// `sentient_for.getBound().getDefiningOp() == query_map`.
    LoopBound,
    /// Every other operand of every other op, including a `sentient.for` operand that is not its
    /// bound.
    Other,
}

/// WHAT `LowerSymbolQueryMap` LEAVES BEHIND.
#[derive(Debug, Clone, PartialEq)]
pub enum QueryMapLowering {
    /// `query_map.replaceAllUsesWith(immutable_mapping.getValues().back())` — every use takes this
    /// value and no conditional is emitted.
    Replaced(Val),
    /// The conditional, and the value each use is re-pointed at, one per entry of the `uses` slice
    /// and in the same order.
    Conditional {
        /// `sentient::IfOp if_op = createIfOpFromMapping(query_map, num_results);`
        if_op: sen::Op,
        /// `owner_op->setOperand(operand_num, if_op.getResults()[..])`, per use.
        replacements: Vec<Val>,
    },
}

/// Replaces: e275_LowerSymbolQueryMap
///
/// **275/384** `SymbolToSentientLoweringPass::LowerSymbolQueryMap` —
/// `dcc/src/Conversion/SymbolToSentient/SymbolToSentient.cpp:40` (68L).
///
/// THE MAPPING AS ONE `sentient.if` CHAIN, AND WHICH OF ITS RESULTS EACH USE READS.
///
/// ⛔ THE RESULT COUNT IS NOT THE USE COUNT: one JCR result for the loop-bound uses TOGETHER, then
/// one LRF result for all non-loop-bound uses — except on L3, which gets one PER non-loop-bound use
/// (`:79-86`).
/// ⛔ `uses` IS IN MLIR USE-LIST ORDER, WHICH IS REVERSE PROGRAM ORDER, and the reference walks it
/// backwards (`:94-95`) — so the L3 result indices ascend in PROGRAM order. `replacements` comes
/// back aligned with `uses`, not with the walk.
#[must_use]
pub fn lower_symbol_query_map(
    comp: GenericComp,
    mapping: SymbolMapping<'_>,
    uses: &[QueryMapUse],
    values: &mut Values,
) -> QueryMapLowering {
    let chain = match mapping {
        // `if (immutable_mapping.getKeys().size() == 1) { .. return; }`
        SymbolMapping::Single(value) => return QueryMapLowering::Replaced(value),
        SymbolMapping::Chain(chain) => chain,
    };

    // The use walk, which only ever asked these two questions of it.
    let found_loop_bound_use = uses.contains(&QueryMapUse::LoopBound);
    let num_non_loop_bound_uses = uses.iter().filter(|u| **u == QueryMapUse::Other).count();

    // `is_any_of(comp, L3LU, L3SU)`.
    let is_l3 = matches!(comp, GenericComp::L3lu | GenericComp::L3su);

    // `int num_results = (found_loop_bound_use ? 1 : 0) + ((num_non_loop_bound_uses > 0) ?
    //  (is_any_of(comp, L3LU, L3SU) ? num_non_loop_bound_uses : 1) : 0);`
    let num_results = usize::from(found_loop_bound_use)
        + if num_non_loop_bound_uses == 0 {
            0
        } else if is_l3 {
            num_non_loop_bound_uses
        } else {
            1
        };

    let if_op = create_if_op_from_mapping(chain, num_results, values);
    // `if_op.getResults()` — the values [`create_if_op_from_mapping`] minted. The `_` arm is
    // unreachable: entry 226 builds a [`sen::Op::If`] and nothing else.
    let results: Vec<Val> = match &if_op {
        sen::Op::If { yielded, .. } => yielded.iter().map(|slot| slot.result).collect(),
        _ => Vec::new(),
    };

    // `int if_op_result_num_to_use_for_replacement = (found_loop_bound_use ? 1 : 0);`
    let mut next = usize::from(found_loop_bound_use);
    let mut replacements: Vec<Val> = Vec::with_capacity(uses.len());
    // `for (auto it = ..rbegin(); it != ..rend(); ++it)`.
    for use_kind in uses.iter().rev() {
        match use_kind {
            // `sentient_for.setOperand(operand_num, if_op.getResults()[0]); continue;`
            QueryMapUse::LoopBound => replacements.extend(results.first().copied()),
            QueryMapUse::Other => {
                replacements.extend(results.get(next).copied());
                // `if (is_any_of(comp, L3LU, L3SU)) ++if_op_result_num_to_use_for_replacement;`
                if is_l3 {
                    next += 1;
                }
            }
        }
    }
    // Back into `uses` order, the walk having run against it in reverse.
    replacements.reverse();

    QueryMapLowering::Conditional {
        if_op,
        replacements,
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 303/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// ONE `symbol.query_map` THE PASS LOWERED, AND THE TWO OPS THAT LEAVES DEAD.
#[derive(Debug, Clone, PartialEq)]
pub struct LoweredQueryMap<'a> {
    /// `to_be_deleted_list.push_back(query_map)` (`SymbolToSentient.cpp:46`).
    pub query_map: &'a DfirOp,
    /// `to_be_deleted_list.push_back(immutable_mapping)` (`:47`) — queued for BOTH shapes, since
    /// both pushes precede the one-pair early return.
    pub mapping: &'a DfirOp,
    /// Entry 275's answer, its `replacements` aligned with `uses` as documented there.
    pub lowering: QueryMapLowering,
}

/// WHAT ONE `dataflow.program_unit` CAME OUT OF THE PASS AS.
#[derive(Debug, Clone, PartialEq)]
pub struct UnitQueryMaps<'a> {
    /// `getUnitType(unit.getUnits()[0].getDefiningOp<GetUnitOp>())` (`:26-27`) — see entry 196 for
    /// why the island reads this off [`crate::islands::dataflow_ir::Units::kind`].
    pub comp: GenericComp,
    /// This unit's query maps in pre-order (`:29-32`), each with its lowering.
    pub lowered: Vec<LoweredQueryMap<'a>>,
}

/// Replaces: e303_runOnOperation
///
/// **303/384** `SymbolToSentientLoweringPass::runOnOperation` —
/// `dcc/src/Conversion/SymbolToSentient/SymbolToSentient.cpp:23` (15L). Entry 275 over every
/// `symbol.query_map` of every program unit, under that unit's own component.
///
/// ⛔ THE ERASE LOOP'S `DT_CHECK_MSG(op->use_empty())` (`:33-35`) HOLDS BY CONSTRUCTION: entry 275
/// re-points every use it was handed, and [`LoweredQueryMap`] carries the pair rather than erasing.
/// ⛔ A query whose `map` is not a `symbol.symbol_immutable_mapping` is the `DT_CHECK` at `:45` — it
/// is skipped here, so it is neither lowered nor queued.
#[must_use]
pub fn run_on_operation<'a, A: Arch>(
    program: &'a Program<A>,
    values: &mut Values,
) -> Vec<UnitQueryMaps<'a>> {
    let mut per_unit: Vec<UnitQueryMaps<'_>> = Vec::new();
    // `module_op.walk<PreOrder>([&](dataflow::ProgramUnitOp unit) { .. })` — the units are a field of
    // the program rather than ops among ops, exactly as entry 196 records.
    for unit in program.units.iter() {
        // `dcc::getUnitType(unit.getUnits()[0].getDefiningOp<GetUnitOp>())` (`:26-27`) — ⭐ THE
        // COMPONENT IS ALREADY GENERIC: `getUnitType` maps through `senCompToGenericComp` itself
        // (`DccExtContext.cpp:130`), so entry 275's `is_any_of(comp, L3LU, L3SU)` (`:85`) compares
        // generic components, not the raw `type` string.
        let comp = unit.on.kind().generic();
        let mut sites: Vec<&DfirOp> = Vec::new();
        query_maps(&unit.body, &mut sites);

        let mut lowered: Vec<LoweredQueryMap<'_>> = Vec::new();
        for site in sites {
            let DfirOp::Symbol(symbol::Op::QueryMap { result, map, key }) = site else {
                continue;
            };
            // `query_map.getMap().getDefiningOp<symbol::SymbolImmutableMappingOp>()`.
            let Some(mapping) = defining_op(*map, &unit.body) else {
                continue;
            };
            let DfirOp::Symbol(symbol::Op::ImmutableMapping { pairs, .. }) = mapping else {
                continue;
            };
            // `immutable_mapping.getKeys().size() == 1` is [`SymbolMapping::Single`]; two or more is
            // the chain. An empty table has no `getValues().back()` and cannot be built by the
            // scheduler, so it is not a site.
            let symbols = match pairs.as_slice() {
                [] => continue,
                [(_, value)] => SymbolMapping::Single(*value),
                [first, .., (_, last_value)] => SymbolMapping::Chain(QueryMapping {
                    key: *key,
                    first: *first,
                    middle: &pairs[1..pairs.len() - 1],
                    last_value: *last_value,
                }),
            };
            lowered.push(LoweredQueryMap {
                query_map: site,
                mapping,
                lowering: lower_symbol_query_map(
                    comp,
                    symbols,
                    &query_map_uses(*result, &unit.body),
                    values,
                ),
            });
        }
        per_unit.push(UnitQueryMaps { comp, lowered });
    }
    per_unit
}

/// `unit.walk<WalkOrder::PreOrder>([&](Operation* op) { if (dyn_cast<SymbolQueryMapOp>(op)) .. })`
/// (`SymbolToSentient.cpp:29-32`) — the op before the ops of its regions, which for this op is only
/// the enclosing loops' order.
fn query_maps<'a>(body: &'a [DfirOp], found: &mut Vec<&'a DfirOp>) {
    for op in body {
        if matches!(op, DfirOp::Symbol(symbol::Op::QueryMap { .. })) {
            found.push(op);
        }
        for region in regions(op) {
            query_maps(region, found);
        }
    }
}

/// EVERY USE OF THE QUERY'S RESULT, CLASSIFIED, IN MLIR USE-LIST ORDER — `for (auto& use :
/// query_map.getResult().getUses())` (`SymbolToSentient.cpp:64-75`).
///
/// ⛔⛔ THE TEST IS ON THE **LOOP**, NOT ON THIS USE'S OPERAND NUMBER: `sentient_for.getBound()
/// .getDefiningOp() == query_map` (`:69`) is asked once per use, so a loop that reads the query BOTH
/// as its bound and as something else has BOTH uses counted as loop-bound uses. Classifying by
/// operand number instead would inflate `num_non_loop_bound_uses` and, on L3, mint a result nothing
/// reads.
/// ⛔ THE BOUND IS THE **UPPER** ONE. `sentient.for %i = %bound` is a trip count and what lowers into
/// it is the DataflowIR loop's `hi`; a use as `lo`, as `step` or as an `iter_args` init is an `Other`.
/// ⛔ AND THE ORDER IS REVERSED, because MLIR's use list is reverse program order (`:91-92`) and
/// entry 275 walks it backwards.
fn query_map_uses(result: Val, body: &[DfirOp]) -> Vec<QueryMapUse> {
    let mut classified: Vec<QueryMapUse> = uses(result, body)
        .into_iter()
        .map(|user| match user {
            DfirOp::Scf(scf::Op::For { hi, .. }) if *hi == result => QueryMapUse::LoopBound,
            DfirOp::Affine(affine::Op::For {
                hi: affine::Bound::Val(hi),
                ..
            }) if *hi == result => QueryMapUse::LoopBound,
            _ => QueryMapUse::Other,
        })
        .collect();
    classified.reverse();
    classified
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Dd2;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::dialects::arith;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::dataflow_ir::{
        Grid, GroupId, OpIndex, ProgramName, ProgramUnit, ProgramUnits, Units,
    };
    use crate::units::DfirUnit;

    /// 🎯 226/384 — THE THREE-PAIR MAPPING OF THE REFERENCE'S OWN WORKED EXAMPLE
    /// (`SymbolToSentient.cpp:112-121`), with two results.
    ///
    /// ⛔ THE LAST PAIR HAS NO COMPARISON: three mappings give TWO `sentient.if`s, and the third
    /// value is what the innermost `else` yields.
    #[test]
    fn a_mapping_becomes_a_right_nested_chain_ending_in_the_last_value() {
        let mut values = Values::default();
        let key = values.mint();
        let (k1, v1) = (values.mint(), values.mint());
        let (k2, v2) = (values.mint(), values.mint());
        let v3 = values.mint();

        let if_op = create_if_op_from_mapping(
            QueryMapping {
                key,
                first: (k1, v1),
                middle: &[(k2, v2)],
                last_value: v3,
            },
            2,
            &mut values,
        );

        let outer = vec![Val(6), Val(7)];
        let inner = vec![Val(8), Val(9)];
        let unknown = sen::Reg {
            locale: sen::RegType::Unknown,
            index: None,
        };
        let yielded = |results: &[Val]| {
            results
                .iter()
                .map(|result| sen::Yielded {
                    result: *result,
                    reg: unknown,
                    element_size: None,
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(
            if_op,
            sen::Op::If {
                predicate: sen::CmpPredicate::Eq,
                lhs: key,
                rhs: k1,
                yielded: yielded(&outer),
                dbg_name: None,
                then_body: vec![SenOp::Sentient(sen::Op::Yield {
                    results: vec![v1, v1]
                })],
                else_body: vec![
                    SenOp::Sentient(sen::Op::If {
                        predicate: sen::CmpPredicate::Eq,
                        lhs: key,
                        rhs: k2,
                        yielded: yielded(&inner),
                        dbg_name: None,
                        then_body: vec![SenOp::Sentient(sen::Op::Yield {
                            results: vec![v2, v2]
                        })],
                        else_body: vec![SenOp::Sentient(sen::Op::Yield {
                            results: vec![v3, v3]
                        })],
                    }),
                    SenOp::Sentient(sen::Op::Yield { results: inner }),
                ],
            }
        );
    }

    /// 🎯 275/384 — THE VENDOR'S OWN `@multiple_uses` AND `@multiple_uses_lx_or_below`
    /// (`dcc/test/Conversion/SymbolToSentient/symbols_query.mlir`), which differ only in `comp`.
    ///
    /// Four pairs, and four uses in program order: two `sentient.for` bounds and two `scalar_sub`
    /// operands. ⛔ L3 GETS `:3` AND BOTH BOUNDS STILL SHARE `#0` — `%12#0, %12#1` then
    /// `%12#0, %12#2`; LX gets `:2` and both subs collapse onto `#1`.
    #[test]
    fn the_l3_case_numbers_each_non_loop_bound_use_and_lx_reuses_one() {
        let case = |comp: GenericComp| {
            let mut values = Values::default();
            let key = values.mint();
            let pairs: Vec<(Val, Val)> = (0..4).map(|_| (values.mint(), values.mint())).collect();
            let mapping = SymbolMapping::Chain(QueryMapping {
                key,
                first: pairs[0],
                middle: &pairs[1..3],
                last_value: pairs[3].1,
            });
            // Use-list order is reverse program order; program order is bound, sub, bound, sub.
            let uses = [
                QueryMapUse::Other,
                QueryMapUse::LoopBound,
                QueryMapUse::Other,
                QueryMapUse::LoopBound,
            ];
            let lowered = lower_symbol_query_map(comp, mapping, &uses, &mut values);
            let QueryMapLowering::Conditional {
                if_op,
                replacements,
            } = lowered
            else {
                unreachable!("four pairs is not the one-pair case")
            };
            let sen::Op::If { yielded, .. } = &if_op else {
                unreachable!("entry 226 emits a conditional")
            };
            let results: Vec<Val> = yielded.iter().map(|slot| slot.result).collect();
            // Back to program order for the comparison.
            let mut in_program_order = replacements;
            in_program_order.reverse();
            (results, in_program_order)
        };

        let (l3, l3_uses) = case(GenericComp::L3su);
        assert_eq!(l3.len(), 3);
        assert_eq!(l3_uses, vec![l3[0], l3[1], l3[0], l3[2]]);

        let (lx, lx_uses) = case(GenericComp::Lxsu);
        assert_eq!(lx.len(), 2);
        assert_eq!(lx_uses, vec![lx[0], lx[1], lx[0], lx[1]]);
    }

    /// 🎯 303/384 — THE VENDOR'S OWN `@basic`
    /// (`dcc/test/Conversion/SymbolToSentient/symbols_query.mlir:128-147`): one `l3su` unit, a
    /// two-pair mapping inside a loop, and a query used as the inner loop's bound AND as a subtract's
    /// operand. Its expectation is `%[[VAL_9]]:2`, with `#0` the bound and `#1` the subtract
    /// (`:19-24`).
    ///
    /// ⛔ THE MAPPING AND THE QUERY ARE BOTH INSIDE THE OUTER LOOP, so a walk that stopped at the
    /// unit's top level would find neither.
    #[test]
    fn the_vendors_basic_case_numbers_the_bound_and_the_subtract_apart() {
        let mut values = Values::default();
        let symbol0 = values.mint();
        let symbol1 = values.mint();
        let (c0, c1, c2) = (values.mint(), values.mint(), values.mint());
        let (arg0, sub, map, query, arg1, sub2) = (
            values.mint(),
            values.mint(),
            values.mint(),
            values.mint(),
            values.mint(),
            values.mint(),
        );
        let subtract = |result, lhs, rhs| {
            DfirOp::Arith(arith::Op::SubI(arith::IntBinary {
                result,
                lhs,
                rhs,
                ty: ScalarTy::Index,
            }))
        };
        // `sentient.for` is a trip count, so the DataflowIR loop it lowers from carries the query as
        // its `hi`.
        let outer_body = vec![
            subtract(sub, c2, arg0),
            DfirOp::Symbol(symbol::Op::ImmutableMapping {
                result: map,
                pairs: vec![(c0, symbol0), (c1, symbol1)],
            }),
            DfirOp::Symbol(symbol::Op::QueryMap {
                result: query,
                map,
                key: sub,
            }),
            DfirOp::Scf(scf::Op::For {
                iv: arg1,
                lo: c0,
                hi: query,
                step: c1,
                carried: Vec::new(),
                body: vec![subtract(sub2, query, arg1)],
                dbg_name: None,
            }),
        ];
        let unit = ProgramUnit {
            on: Units::one(DfirUnit::L3su, Val(1)),
            precision: None,
            body: vec![DfirOp::Scf(scf::Op::For {
                iv: arg0,
                lo: c0,
                hi: c2,
                step: c1,
                carried: Vec::new(),
                body: outer_body,
                dbg_name: None,
            })],
            arch: core::marker::PhantomData,
        };
        let program: Program<Dd2> = Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            grid: Grid::single(),
            preamble: Vec::new(),
            units: ProgramUnits::of(unit, Vec::new()),
            arch: core::marker::PhantomData,
        };

        let per_unit = run_on_operation(&program, &mut values);
        let [unit] = per_unit.as_slice() else {
            panic!("one program unit")
        };
        assert_eq!(unit.comp, GenericComp::L3su);
        let [site] = unit.lowered.as_slice() else {
            panic!("one query map, found inside the outer loop")
        };
        // The pair the erase loop walks, in the order it was queued.
        assert!(matches!(
            site.query_map,
            DfirOp::Symbol(symbol::Op::QueryMap { .. })
        ));
        assert!(matches!(
            site.mapping,
            DfirOp::Symbol(symbol::Op::ImmutableMapping { .. })
        ));
        let QueryMapLowering::Conditional {
            if_op,
            replacements,
        } = &site.lowering
        else {
            panic!("two pairs is the chain")
        };
        let sen::Op::If { yielded, .. } = if_op else {
            panic!("entry 226 emits a conditional")
        };
        let results: Vec<Val> = yielded.iter().map(|slot| slot.result).collect();
        assert_eq!(results.len(), 2);
        // `replacements` is in use-list order; program order is the loop bound then the subtract.
        let mut in_program_order = replacements.clone();
        in_program_order.reverse();
        assert_eq!(in_program_order, vec![results[0], results[1]]);
    }
}
