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

//! `OldRegisterInitialization.cpp` — 10 of the campaign's 656 units (dependency level(s) [0, 1, 4, 6, 7]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e102_dumpWeights` | 102 | 0 | 6 | `dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:145` |
//! | `e108_removeInitAttrFromOps` | 108 | 0 | 19 | `dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:686` |
//! | `e109_eraseDeletedOps` | 109 | 0 | 4 | `dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:768` |
//! | `e110_hasSameAttr` | 110 | 0 | 7 | `dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:774` |
//! | `e326_replaceVirtualAssignTarget` | 326 | 1 | 8 | `dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:159` |
//! | `e331_dumpWeight` | 331 | 1 | 5 | `dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:706` |
//! | `e332_moveSSAToInit` | 332 | 1 | 25 | `dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:715` |
//! | `e570_getFirstSource` | 570 | 4 | 5 | `dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:788` |
//! | `e631_promoteRegisterInitCandidatesAboveUniformRegion` | 631 | 6 | 5 | `dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:1036` |
//! | `e644_runOnOperation` | 644 | 7 | 111 | `dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:1203` |

pub(crate) mod register_init_candidate_promoter;
pub(crate) mod register_init_info;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use crate::arch::Arch;
use crate::formats::Bits;
use crate::islands::dataflow_ir::Values;
use crate::islands::sentient::{Program, ProgramUnit};
use crate::islands::sentient::dialects::{
    self as dialects, Definitions, LocalRegion, Op, Val, regions_ref, sentient,
};
use crate::islands::sentient::print;
use crate::model::Model;
use crate::transform::sentient::analyses::{
    InstructionEstimator, Liveness, RegisterGraphs, UniformGroups,
};
use crate::transform::sentient::utils::units_and_their_values::UnitsAndTheirValues;
use crate::workload::Workload;

use self::register_init_info::{
    EarlyStop, Enclosing, GtrRegInit, RegisterInitInfo, SsaWeights, has_uniformize_region,
};

/// Replaces: e102_dumpWeights
///
/// Every weighted value and its weight, one `--t> N` line each.
///
/// ⛔ TRAP: `llvm::errs()` IS A DUMP AND A DUMP IS A STRING HERE — the caller (`e331_dumpWeight`)
/// wraps these lines in its own banner, so returning them is what lets it.
/// ⭐ ORDERED, WHERE `DenseMap` IS NOT: the reference's iteration order is unspecified, so
/// [`SsaWeights`]' `BTreeMap` is the only version of this whose output is the same twice.
/// ⚠️ A VALUE WITH NO DEFINING OP IN `scope` GETS `Value::print`'S BLOCK-ARGUMENT LINE WITHOUT ITS
/// TYPE — this island carries no per-value type. And [`print::emit`] ends its line, so `--t>` starts
/// the next one rather than following on the same one.
#[must_use]
pub fn dump_weights(ssa_weight: &SsaWeights, scope: &[Op]) -> String {
    let mut out = String::new();
    for (val, weight) in ssa_weight.iter() {
        match dialects::defining_op(val, scope) {
            Some(op) => print::emit(&mut out, op, 0),
            None => {
                let _ = writeln!(out, "<block argument> {val:?}");
            }
        }
        let _ = writeln!(out, "--t> {}", weight.0);
    }
    out
}

/// Replaces: e108_removeInitAttrFromOps
///
/// Clears `programHeader` everywhere under `parent` — every `sentient.scalar_copy` to false, and
/// every `sentient.for`'s whole array to falses.
///
/// ⭐ A SLICE BECAUSE THE WALK INCLUDES THE OP IT STARTS FROM, and one caller starts it at a
/// `scalar_copy` (`:982`) while the other starts it at the program unit (`:1216`).
/// ⛔ TRAP: BOTH OF THE REFERENCE'S GUARDS ARE NO-OPS AT THIS ISLAND, NOT DROPPED WORK. `hasAttr`
/// and `program_headers.empty()` distinguish an absent attribute from an all-false one, and here
/// absent IS all-false — [`sentient::Carried::program_header`] is a `bool` per position and the
/// printer omits the array entirely when none is set, so writing falses over either is the same IR.
pub fn remove_init_attr_from_ops(parent: &mut [Op]) {
    for op in parent.iter_mut() {
        match op {
            Op::Sentient(inner) => {
                match inner {
                    sentient::Op::ScalarCopy { program_header, .. } => *program_header = false,
                    // `numIter = for_op.getNumIterOperands()` falses — one per carried value, which
                    // is this island's array length by construction.
                    sentient::Op::For { carried, .. } => {
                        for position in carried.iter_mut() {
                            position.program_header = false;
                        }
                    }
                    _ => {}
                }
                for region in sentient::regions_mut(inner) {
                    remove_init_attr_from_ops(region);
                }
            }
            Op::AffineFor(loop_op) => remove_init_attr_from_ops(&mut loop_op.body),
            // ⛔ PROVED ABSENT BY THE TYPE: a lower-rung region holds
            // `crate::islands::dataflow_ir::dialects::Op`, and neither `sentient.scalar_copy` nor
            // `sentient.for` is one.
            _ => {}
        }
    }
}

/// Replaces: e109_eraseDeletedOps
///
/// Erases every op the promoter marked and empties the set.
///
/// ⛔ TRAP: `op->erase()` ABORTS MLIR WHILE A USE REMAINS, and clearing the uses is the CALLER'S
/// work, not this function's. ONE of the three insertion sites proves the state with
/// `DT_CHECK(copy_op.getUses().empty())` (`:1030-1031`); the other two reach it by rewiring first,
/// `replaceAllUsesWith` (`:1178-1179`, `:1194-1198`). This only carries the erasure out.
/// ⭐ ORDERED, WHERE `SmallSet` IS NOT: erasing is order-independent, and a set with an order is the
/// version whose effect is the same twice.
pub fn erase_deleted_ops(body: &mut Vec<Op>, to_be_erased: &mut BTreeSet<Val>) {
    for val in to_be_erased.iter() {
        dialects::erase_defining_op(body, *val);
    }
    to_be_erased.clear();
}

/// `dcc::utils::getAttr(val, "element_size")` (`dcc/src/Utils/Utils.cpp:392-407`) — the width the op
/// behind a value declares, or `None` where it declares none.
///
/// ⛔ A LOOP ITER ARGUMENT READS THE LOOP'S `element_sizes` AT ITS OWN POSITION, and this island's
/// array is carried-indexed while the reference's is `[bound, initArgs.., results..]` — so the
/// argument's slot is its [`sentient::Carried`], the same collapse the loop's `regLocales` already
/// gets. The reference's other block-argument index, the induction variable's `element_sizes[0]`, is
/// unreachable here: only `getRegionIterArgs()` values are ever marked (`moveSSAToInit`, `:723-735`).
fn element_size(val: Val, scope: &[Op]) -> Option<Bits> {
    if let Some(carried) = carried_arg(val, scope) {
        return carried.element_size;
    }
    match dialects::defining_op(val, scope)? {
        Op::Sentient(
            sentient::Op::ScalarAdd { element_size, .. }
            | sentient::Op::ScalarSub { element_size, .. }
            | sentient::Op::ScalarCopy { element_size, .. },
        ) => *element_size,
        // The `.td`-DECLARED ones, which are never absent — `$element_size` on the transfers
        // (`SentientOps.td:715`) and on `load_and_extract_scalar` (`:606`).
        Op::Sentient(
            sentient::Op::Load { extent, .. }
            | sentient::Op::LoadAndSend { extent, .. }
            | sentient::Op::ReceiveAndStore { extent, .. }
            | sentient::Op::LoadAndStore { extent, .. },
        ) => Some(extent.element_size),
        Op::Sentient(sentient::Op::LoadAndExtractScalar { element_size, .. }) => {
            Some(*element_size)
        }
        _ => None,
    }
}

/// The `sentient.for` position a block argument belongs to, at any depth of `scope`.
fn carried_arg(val: Val, scope: &[Op]) -> Option<&sentient::Carried> {
    for op in scope {
        if let Op::Sentient(inner) = op {
            if let sentient::Op::For { carried, .. } = inner
                && let Some(found) = carried.iter().find(|position| position.arg == val)
            {
                return Some(found);
            }
            for region in sentient::regions(inner) {
                if let Some(found) = carried_arg(val, region) {
                    return Some(found);
                }
            }
        } else if let Op::AffineFor(loop_op) = op
            && let Some(found) = carried_arg(val, &loop_op.body)
        {
            return Some(found);
        }
    }
    None
}

/// Replaces: e110_hasSameAttr
///
/// Whether two values address elements of the same width — and `true` for a pair where either side
/// is null, which is the reference's own answer for one.
#[must_use]
pub fn has_same_attr(val1: Option<Val>, val2: Option<Val>, scope: &[Op]) -> bool {
    match (val1, val2) {
        (Some(val1), Some(val2)) => element_size(val1, scope) == element_size(val2, scope),
        _ => true,
    }
}

// e326_replaceVirtualAssignTarget is ported ONE MODULE DOWN, as the method it is: it mutates
// `enforced_virtual_assign_` and `final_reg_coalescing_candidates_`, two fields of the
// `RegisterInitInfo` that lives in `register_init_info.rs` beside the four other units of the same
// class this batch fills. A free function here would need both fields passed in and would leave the
// class's state split across two files — see the same promotion at `utils/mod.rs:1214`.
// ⭐ FILLED ANCHOR: `register_init_info.rs`, `RegisterInitInfo::replace_virtual_assign_target`.

/// Replaces: e331_dumpWeight
///
/// Every program unit's weight table between one pair of banners (`OldRegisterInitialization.cpp:706-712`).
///
/// ⛔ TRAP: THE BANNERS EACH OPEN WITH A BLANK LINE, `"\n------Op Weights Begin------\n"`, so the
/// dump is separated from whatever `llvm::errs()` last wrote and again from the first weight line.
#[must_use]
pub fn dump_weight(rtis: &[RegisterInitInfo], scope: &[Op]) -> String {
    let mut out = String::from("\n------Op Weights Begin------\n");
    for rti in rtis {
        out.push_str(&dump_weights(&rti.ssa_weight, scope));
    }
    out.push_str("\n------Op Weights End------\n");
    out
}

/// Replaces: e332_moveSSAToInit
///
/// Marks the op behind one promoted value for the program header: a `sentient.scalar_copy` outright,
/// or the one carried position of a `sentient.for` that binds it (`OldRegisterInitialization.cpp:715-739`).
///
/// ⛔ TRAP: ONLY THE MATCHING LOOP POSITION IS SET AND EVERY OTHER IS PRESERVED. The reference rebuilds
/// the whole `programHeader` array to do it, padding positions the old array never reached with
/// `false` — which is the same IR here, because absent IS all-false; see `e108`'s TRAP.
/// ⚠️ A VALUE THAT IS NEITHER — a loop RESULT, an induction variable, any other op's result — is a
/// no-op, which is the reference's own outcome: its rebuilt array reproduces itself.
pub fn move_ssa_to_init(scope: &mut [Op], val: Val) {
    for op in scope.iter_mut() {
        match op {
            Op::Sentient(inner) => {
                match inner {
                    sentient::Op::ScalarCopy {
                        result,
                        program_header,
                        ..
                    } if *result == val => *program_header = true,
                    sentient::Op::For { carried, .. } => {
                        for position in carried.iter_mut() {
                            if position.arg == val {
                                position.program_header = true;
                            }
                        }
                    }
                    _ => {}
                }
                for region in sentient::regions_mut(inner) {
                    move_ssa_to_init(region, val);
                }
            }
            Op::AffineFor(loop_op) => move_ssa_to_init(&mut loop_op.body, val),
            // ⛔ PROVED ABSENT BY THE TYPE, exactly as `e108`'s tail is.
            _ => {}
        }
    }
}

/// Replaces: e570_getFirstSource
///
/// The value [`register_init_candidate_promoter::get_source`] pairs with the FIRST unit — the one
/// source that stands for all of them once the clones are in place.
///
/// ⛔ TRAP: THE CLONING IS NOT DISCARDED WITH THE LIST. `tmp_uvs` is local, but `getSource` clones
/// each source to the top of the unit body and records erasures as it goes, and those survive.
/// ⛔ `values().front()` ON AN EMPTY LIST IS A DANGLING READ; `None` is that and the reference's own
/// null first value, which `normalizeNullValues` leaves in place for an all-null list.
pub fn get_first_source<U: UniformGroups>(
    unit: &mut Vec<Op>,
    core: Val,
    val: Val,
    to_be_erased: &mut BTreeSet<Val>,
    values: &mut Values,
    uga: &U,
) -> Option<Val> {
    let mut tmp_uvs = UnitsAndTheirValues::default();
    register_init_candidate_promoter::get_source(
        unit,
        core,
        val,
        &mut tmp_uvs,
        to_be_erased,
        values,
        uga,
    );
    tmp_uvs.pairs.first().and_then(|(_unit, value)| *value)
}

/// Replaces: e631_promoteRegisterInitCandidatesAboveUniformRegion
///
/// Runs [`register_init_candidate_promoter::run`] over one unit — the whole body is constructing the
/// promoter and calling it.
///
/// ⛔ THE PROMOTER'S THREE STORAGE MEMBERS ARE FRESH PER CALL (`:798-800`): the candidates, the
/// erasures and the replacement pairs die with the promoter, so they are locals here and NOT the
/// pass's own state, which `runOnOperation` keeps separately.
/// ⭐ `arg` IS A PARAMETER BECAUSE THE ISLAND CARRIES NO `iter_arg` — the recorded gap
/// [`crate::islands::sentient::dialects::Definitions::within_program_unit`] names, and the same
/// mechanism-supplied-by-the-caller decision.
pub fn promote_register_init_candidates_above_uniform_region<
    A: Arch,
    L: Liveness + Clone,
    G: RegisterGraphs + Default,
    U: UniformGroups,
>(
    unit: &mut ProgramUnit<A>,
    arg: Val,
    base_fold_units: &[Val],
    liveness: &mut L,
    rtis: &mut BTreeMap<Val, RegisterInitInfo>,
    uga: &U,
    values: &mut Values,
) {
    let on = unit.on.kind();
    let units = unit.on.vals();
    let mut to_be_erased = BTreeSet::new();
    let mut all_reginit_candidates = Vec::new();
    let mut all_replacement_pairs = Vec::new();
    register_init_candidate_promoter::run::<A, L, G, U>(
        &mut unit.body,
        on,
        arg,
        &units,
        base_fold_units,
        rtis,
        liveness,
        uga,
        values,
        &mut to_be_erased,
        &mut all_reginit_candidates,
        &mut all_replacement_pairs,
    );
}

/// `DisableThisPass` — `-dcc-old-register-initialization-disable`, `cl::init(false)` (`:51-54`).
const DISABLE_THIS_PASS: bool = false;

/// `DoPromoteAboveUniform` — `cl::init(true)` (`:61-65`).
const DO_PROMOTE_ABOVE_UNIFORM: bool = true;

/// `DoUniformGroupLeadersOnly` — `cl::init(true)` (`:75-79`).
const DO_UNIFORM_GROUP_LEADERS_ONLY: bool = true;

/// `opts_.OptLevel == 0` (`:1211`) — the pipeline runs this pass above level zero, so the
/// IBUFF-space bail-out is unreachable and the child estimator is never asked.
const OPT_LEVEL_ZERO: bool = false;

/// `UniformizeRegionsOp::getRegionFromUnit` and its `EqualizePatternOp` twin
/// (`dataflow-scheduler/.../lib/Dialect/Uniform/Uniform.cpp:215-229`, `:391`).
///
/// ⭐ "THE FIRST REGION WHOSE UNIT LIST HOLDS `u`" IS THE WHOLE REFERENCE: it indexes the flat unit
/// list, then walks `list_sizes` until the running total passes that index — and a unit the op does
/// not name leaves the index at the total, which no partial sum exceeds, hence `None`.
fn region_from_unit(regions: &[LocalRegion], u: Val) -> Option<&LocalRegion> {
    regions.iter().find(|region| region.units.contains(&u))
}

/// `CollectAndFinalizeRegInitAndCoalescingCandidates`' walk (`:1235-1261`) — every op of the program
/// unit scored and collected for ONE unit value, a `uniform` op contributing the one region that unit
/// runs and its subtree skipped with it.
#[allow(clippy::too_many_arguments)]
fn walk_for_unit<'a, L: Liveness>(
    region: &'a [Op],
    u: Val,
    outer_ops: &[&'a Op],
    outer_scopes: &[&'a [Op]],
    preamble: &'a [Op],
    prog_unit: (Val, &'a [Val]),
    liveness: &L,
    rti: &mut RegisterInitInfo,
) {
    let mut scopes: Vec<&'a [Op]> = Vec::with_capacity(outer_scopes.len() + 1);
    scopes.push(region);
    scopes.extend_from_slice(outer_scopes);
    // ⭐ TWO CHAINS, BECAUSE `getDefiningOp()` IS GLOBAL AND `hasOneUse()` IS TOO: the preamble's
    // `dataflow.get_unit`s and constants must be visible to a lookup, while the scope `e329` counts
    // uses in is the one that holds them, which is the unit body.
    let mut global: Vec<&'a [Op]> = scopes.clone();
    global.push(preamble);
    let use_scope: &[Op] = outer_scopes.last().copied().unwrap_or(region);
    for op in region {
        let enclosing = Enclosing::from_innermost(outer_ops);
        let defs = Definitions::within_program_unit(&global, prog_unit.0, prog_unit.1);
        if let Op::UniformRegions(uniform) = op {
            if let Some(local) = region_from_unit(uniform.regions(), u) {
                rti.calc_ssa_weight_in_region(
                    &local.body,
                    enclosing,
                    &global,
                    GtrRegInit::default(),
                );
                rti.collect_all_reg_coalescing_candidates_in_region(&local.body, &scopes, liveness);
            }
            continue;
        }
        rti.calc_ssa_weight(op, enclosing, defs, GtrRegInit::default());
        rti.collect_all_reg_coalescing_candidates(op, use_scope, defs, liveness);
        let mut inner_ops: Vec<&'a Op> = Vec::with_capacity(outer_ops.len() + 1);
        inner_ops.push(op);
        inner_ops.extend_from_slice(outer_ops);
        for inner in regions_ref(op) {
            walk_for_unit(
                inner, u, &inner_ops, &scopes, preamble, prog_unit, liveness, rti,
            );
        }
    }
}

/// The rest of `CollectAndFinalizeRegInitAndCoalescingCandidates` (`:1262-1265`) — the scoreboard one
/// unit value earned, sorted and then vetted.
#[allow(clippy::too_many_arguments)]
fn collect_and_finalize<A: Arch, L: Liveness + Clone, G: RegisterGraphs + Default>(
    unit_body: &[Op],
    preamble: &[Op],
    prog_unit_arg: Val,
    unit_units: &[Val],
    u: Val,
    liveness: &L,
    rtis: &mut BTreeMap<Val, RegisterInitInfo>,
) {
    let mut rti = RegisterInitInfo::default();
    let none: [&Op; 0] = [];
    walk_for_unit(
        unit_body,
        u,
        &none,
        &[],
        preamble,
        (prog_unit_arg, unit_units),
        liveness,
        &mut rti,
    );
    rti.sort_reg_coalescing_candidates();
    let global: [&[Op]; 2] = [unit_body, preamble];
    let defs = Definitions::within_program_unit(&global, prog_unit_arg, unit_units);
    rti.collect_reg_init_and_reg_coalescing_candidate_fast::<A, L, G>(
        unit_body,
        u,
        defs,
        liveness,
        EarlyStop::default(),
    );
    rtis.insert(u, rti);
}

/// Replaces: e644_runOnOperation
///
/// THE PASS ENTRY (`:1203-1313`): every program unit loses its old header flags and then earns one
/// scoreboard per unit value, which is spent by promoting the candidates above the uniform region
/// when the unit has one and by moving them into the program header when it has not.
///
/// ⛔ THE `cached_rtis` MEMO CAN NEVER HIT: `base_fold_units` already deduplicates on
/// `get_unit_op->getResult(0)`, which is the memo's own key, so it is dropped as mechanism.
/// ⛔ `prog_unit_args` IS THE DROPPED MECHANISM, one per unit of `program.units` in order — no
/// [`ProgramUnit`] carries its body's argument; see [`super::deuniform::run_on_operation`].
/// ⛔ `markAnalysesPreserved<Liveness>` (`:1312`) is pass-manager bookkeeping and is dropped.
#[allow(clippy::too_many_arguments)]
pub fn run_on_operation<
    A: Arch,
    M: Model,
    W: Workload,
    I: InstructionEstimator,
    L: Liveness + Clone,
    G: RegisterGraphs + Default,
    U: UniformGroups,
>(
    program: &mut Program<A, M, W>,
    prog_unit_args: &[Val],
    estimator: &mut I,
    liveness: &mut L,
    uga: &mut U,
    values: &mut Values,
) {
    if DISABLE_THIS_PASS {
        return;
    }
    let Program {
        preamble, units, ..
    } = program;
    for (index, unit) in units.iter_mut().enumerate() {
        let Some(prog_unit_arg) = prog_unit_args.get(index).copied() else {
            todo!(
                "OldRegisterInitializationPass::runOnOperation: no program-unit argument for unit \
                 {index}, which `unit_.getRegion().getArguments()[0]` is (:934)"
            )
        };
        if OPT_LEVEL_ZERO && estimator.have_ibuff_space(&unit.body) {
            continue;
        }
        // "always remove existing programheader attr" (`:1215-1216`).
        remove_init_attr_from_ops(&mut unit.body);
        let mut rtis: BTreeMap<Val, RegisterInitInfo> = BTreeMap::new();
        let unit_units = unit.on.vals();
        if has_uniformize_region(&unit.body) {
            if DO_UNIFORM_GROUP_LEADERS_ONLY {
                uga.collect_exclusive_group_leaders();
            }
            let units_ref = if DO_UNIFORM_GROUP_LEADERS_ONLY {
                uga.group_leaders()
            } else {
                unit_units.clone()
            };
            let mut base_fold_units: Vec<Val> = Vec::new();
            for u in units_ref {
                // `get_unit_op->getResult(0)` IS `u`: a `dataflow.get_unit` binds exactly one result.
                if !base_fold_units.contains(&u) {
                    collect_and_finalize::<A, L, G>(
                        &unit.body,
                        preamble,
                        prog_unit_arg,
                        &unit_units,
                        u,
                        liveness,
                        &mut rtis,
                    );
                    base_fold_units.push(u);
                }
            }
            if DO_PROMOTE_ABOVE_UNIFORM {
                promote_register_init_candidates_above_uniform_region::<A, L, G, U>(
                    unit,
                    prog_unit_arg,
                    &base_fold_units,
                    liveness,
                    &mut rtis,
                    uga,
                    values,
                );
            }
        } else {
            // "All unit share the code so any one arg is ok in this case" (`:1294-1295`).
            collect_and_finalize::<A, L, G>(
                &unit.body,
                preamble,
                prog_unit_arg,
                &unit_units,
                unit.on.first(),
                liveness,
                &mut rtis,
            );
            for rti in rtis.values_mut() {
                // `final_reginit_candidates_` IS A `std::stack`, so `top()` is the LAST entry.
                while let Some(val) = rti.final_reginit_candidates.pop() {
                    liveness.update_live_ranges_for_program_header_promotion(val);
                    move_ssa_to_init(&mut unit.body, val);
                }
                liveness.add_virtual_assign_optional(&rti.final_reg_coalescing_candidates);
                liveness.add_virtual_assign_enforced(&rti.enforced_virtual_assign);
            }
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::register_init_info::{RegisterInitInfo, SsaWeight, SsaWeights};
    use super::{
        dump_weight, dump_weights, erase_deleted_ops, get_first_source, has_same_attr,
        move_ssa_to_init, promote_register_init_candidates_above_uniform_region,
        remove_init_attr_from_ops, run_on_operation,
    };
    use crate::arch::Dd2;
    use crate::formats::Bits;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::dataflow_ir::{GroupId, OpIndex, ProgramName, Units, Values};
    use crate::islands::sentient::dialects::{Op, Val, dataflow, sentient};
    use crate::islands::sentient::{Program, ProgramUnit, ProgramUnits};
    use crate::model::Model;
    use crate::transform::sentient::analyses::{
        Liveness, OutOfScopeInstructionEstimator, OutOfScopeRegisterGraphs, UniformGroups,
        VirtualAssigns,
    };
    use crate::units::{DfirUnit, Residency};
    use crate::workload::Workload;
    use std::collections::{BTreeMap, BTreeSet};

    /// The out-of-scope group analysis, answering for ONE leader with no followers.
    struct OneUnit(Val);

    impl UniformGroups for OneUnit {
        fn group_leaders(&self) -> Vec<Val> {
            vec![self.0]
        }

        fn is_group_leader(&self, unit: Val) -> bool {
            unit == self.0
        }

        fn group_members_led_by(&self, _leader: Val) -> Vec<Val> {
            Vec::new()
        }
    }

    /// e570 — the first (and here only) pair's value is the candidate's source; a candidate that is
    /// not a `sentient.scalar_copy` pairs nothing, which is `values().front()` of an empty list.
    #[test]
    fn e570_takes_the_first_paired_source_and_nothing_from_a_non_copy() {
        let mut unit = vec![
            Op::Dataflow(dataflow::Op::GetUnit {
                result: Val(1),
                residency: Residency::Global,
                unit: DfirUnit::L3lu,
                num_folds: None,
                reg_locale: None,
            }),
            // The source `%7` is defined outside this unit, so nothing is cloned or erased.
            copy(7, 8, None, false),
        ];
        let uga = OneUnit(Val(1));
        let mut to_be_erased = BTreeSet::new();
        let mut values = Values::default();
        for _ in 0..9 {
            let _ = values.mint();
        }
        assert_eq!(
            get_first_source(&mut unit, Val(1), Val(8), &mut to_be_erased, &mut values, &uga),
            Some(Val(7))
        );
        assert!(to_be_erased.is_empty());
        assert_eq!(
            get_first_source(&mut unit, Val(1), Val(1), &mut to_be_erased, &mut values, &uga),
            None
        );
    }

    /// `%r = sentient.scalar_copy %in`, in or out of the program header and at a width.
    fn copy(input: u32, result: u32, element_size: Option<Bits>, program_header: bool) -> Op {
        Op::Sentient(sentient::Op::ScalarCopy {
            input: Val(input),
            result: Val(result),
            reg: sentient::Reg {
                locale: sentient::RegType::Lbr,
                index: None,
            },
            element_size,
            program_header,
        })
    }

    /// One carried position at a width, in or out of the program header.
    fn carried(init: u32, arg: u32, result: u32, element_size: Option<Bits>) -> sentient::Carried {
        sentient::Carried {
            init: Val(init),
            arg: Val(arg),
            result: Val(result),
            reg: sentient::Reg {
                locale: sentient::RegType::Lbr,
                index: None,
            },
            element_size,
            program_header: true,
        }
    }

    /// e102 — the value's own op, then its weight, in key order.
    #[test]
    fn e102_prints_each_weighted_value_above_its_weight() {
        let scope = vec![copy(1, 2, None, false)];
        let mut weights = SsaWeights::default();
        weights.set(Val(2), SsaWeight(7));
        weights.set(Val(9), SsaWeight(-1));
        assert_eq!(
            dump_weights(&weights, &scope),
            "%2 = sentient.scalar_copy %1  {reg_locale = \"lbr\"} : index\n--t> 7\n<block argument> Val(9)\n--t> -1\n"
        );
    }

    /// e331 — one pair of banners around EVERY unit's table, each opening with a blank line.
    #[test]
    fn e331_wraps_every_units_weights_in_one_pair_of_banners() {
        let scope = vec![copy(1, 2, None, false)];
        let mut first = RegisterInitInfo::default();
        first.ssa_weight.set(Val(2), SsaWeight(7));
        let mut second = RegisterInitInfo::default();
        second.ssa_weight.set(Val(9), SsaWeight(3));

        assert_eq!(
            dump_weight(&[first, second], &scope),
            "\n------Op Weights Begin------\n\
             %2 = sentient.scalar_copy %1  {reg_locale = \"lbr\"} : index\n--t> 7\n\
             <block argument> Val(9)\n--t> 3\n\
             \n------Op Weights End------\n"
        );
    }

    /// e332 — the copy is marked, the loop position that binds the value is marked, and its
    /// neighbour keeps the flag it already had.
    #[test]
    fn e332_marks_only_the_op_behind_the_promoted_value() {
        let mut body = vec![
            copy(1, 2, None, false),
            copy(1, 3, None, false),
            Op::Sentient(sentient::Op::For {
                iv: Val(4),
                bound: Val(5),
                bound_reg: None,
                carried: vec![
                    sentient::Carried {
                        program_header: false,
                        ..carried(1, 6, 7, None)
                    },
                    sentient::Carried {
                        program_header: false,
                        ..carried(1, 8, 9, None)
                    },
                ],
                dbg_name: None,
                body: Vec::new(),
            }),
        ];

        move_ssa_to_init(&mut body, Val(2));
        move_ssa_to_init(&mut body, Val(8));
        // ⚠️ A LOOP RESULT IS A NO-OP, and so is a value nothing here binds.
        move_ssa_to_init(&mut body, Val(7));
        move_ssa_to_init(&mut body, Val(99));

        assert_eq!(
            body,
            vec![
                copy(1, 2, None, true),
                copy(1, 3, None, false),
                Op::Sentient(sentient::Op::For {
                    iv: Val(4),
                    bound: Val(5),
                    bound_reg: None,
                    carried: vec![
                        sentient::Carried {
                            program_header: false,
                            ..carried(1, 6, 7, None)
                        },
                        sentient::Carried {
                            program_header: true,
                            ..carried(1, 8, 9, None)
                        },
                    ],
                    dbg_name: None,
                    body: Vec::new(),
                }),
            ]
        );
    }

    /// e108 — a copy and a loop position both lose the flag, at any depth.
    #[test]
    fn e108_clears_the_program_header_flag_everywhere() {
        let mut body = vec![Op::Sentient(sentient::Op::For {
            iv: Val(3),
            bound: Val(4),
            bound_reg: None,
            carried: vec![carried(1, 5, 6, None)],
            dbg_name: None,
            body: vec![copy(5, 7, None, true)],
        })];
        remove_init_attr_from_ops(&mut body);
        assert_eq!(
            body,
            vec![Op::Sentient(sentient::Op::For {
                iv: Val(3),
                bound: Val(4),
                bound_reg: None,
                carried: vec![sentient::Carried {
                    program_header: false,
                    ..carried(1, 5, 6, None)
                }],
                dbg_name: None,
                body: vec![copy(5, 7, None, false)],
            })]
        );
    }

    /// e109 — the marked ops go, the unmarked one stays, and the set is empty afterwards.
    #[test]
    fn e109_erases_every_marked_op_and_empties_the_set() {
        let mut body = vec![copy(1, 2, None, false), copy(1, 3, None, false)];
        let mut to_be_erased = BTreeSet::from([Val(2)]);
        erase_deleted_ops(&mut body, &mut to_be_erased);
        assert_eq!(body, vec![copy(1, 3, None, false)]);
        assert!(to_be_erased.is_empty());
    }

    /// e110 — equal widths agree, a differing pair does not, two silent ops agree, and a null side
    /// agrees with anything.
    #[test]
    fn e110_compares_element_sizes_and_says_yes_to_a_null_side() {
        let scope = vec![
            copy(1, 2, Some(Bits(8)), false),
            copy(1, 3, Some(Bits(8)), false),
            copy(1, 4, Some(Bits(16)), false),
            Op::Sentient(sentient::Op::ScalarAdd {
                lhs: Val(1),
                rhs: Val(1),
                result: Val(5),
                reg: None,
                element_size: None,
                ty: ScalarTy::Index,
            }),
            Op::Sentient(sentient::Op::For {
                iv: Val(6),
                bound: Val(7),
                bound_reg: None,
                carried: vec![carried(2, 8, 9, Some(Bits(16)))],
                dbg_name: None,
                body: Vec::new(),
            }),
        ];
        assert!(has_same_attr(Some(Val(2)), Some(Val(3)), &scope));
        assert!(!has_same_attr(Some(Val(2)), Some(Val(4)), &scope));
        // Two ops that carry no width at all agree, as two null attributes do.
        assert!(has_same_attr(Some(Val(5)), None, &scope));
        assert!(has_same_attr(None, Some(Val(2)), &scope));
        // ⭐ AND A LOOP ITER ARGUMENT READS THE LOOP'S OWN ARRAY AT ITS POSITION.
        assert!(has_same_attr(Some(Val(8)), Some(Val(4)), &scope));
        assert!(!has_same_attr(Some(Val(8)), Some(Val(2)), &scope));
    }

    /// The out-of-scope liveness, recording what the promoter's post-processing and e644's spending of
    /// a scoreboard ask of it.
    #[derive(Clone, Default)]
    struct Recording {
        cleared: Vec<VirtualAssigns>,
        recomputed: Vec<usize>,
        promoted: Vec<Val>,
        optional: Vec<Vec<Vec<Val>>>,
        enforced: Vec<Vec<(Val, Val)>>,
    }

    impl Liveness for Recording {
        fn update_live_ranges_for_program_header_promotion(&mut self, candidate: Val) {
            self.promoted.push(candidate);
        }

        fn is_live_range_overlaps(&self, _val1: Val, _val2: Val) -> bool {
            todo!("this fake is never asked about an overlap")
        }

        fn clear(&mut self, virtual_assigns: VirtualAssigns) {
            self.cleared.push(virtual_assigns);
        }

        fn compute_register_live_range(&mut self, unit: &[Op]) {
            self.recomputed.push(unit.len());
        }

        fn add_virtual_assign_optional(&mut self, set_of_subsets: &[Vec<Val>]) {
            self.optional.push(set_of_subsets.to_vec());
        }

        fn add_virtual_assign_enforced(&mut self, set_of_pairs: &[(Val, Val)]) {
            self.enforced.push(set_of_pairs.to_vec());
        }
        fn operand_to_index(
            &mut self,
            _value: Val,
        ) -> crate::transform::sentient::analyses::RegNode {
        todo!("no unit here reads a colouring node through this fake")
    }
}

    /// e631 — the promoter runs over the unit's own body and units, and its post-processing lands on
    /// the caller's liveness even when there is no core to take a candidate from.
    #[test]
    fn e631_runs_the_promoter_over_the_unit_and_keeps_none_of_its_state() {
        let mut unit = ProgramUnit::<Dd2> {
            on: Units::one(DfirUnit::L3lu, Val(1)),
            precision: None,
            body: vec![copy(7, 8, None, false)],
            arch: core::marker::PhantomData,
        };
        let mut liveness = Recording::default();
        let mut rtis = BTreeMap::new();
        let uga = OneUnit(Val(1));
        let mut values = Values::default();
        promote_register_init_candidates_above_uniform_region::<Dd2, _, OutOfScopeRegisterGraphs, _>(
            &mut unit,
            Val(0),
            &[],
            &mut liveness,
            &mut rtis,
            &uga,
            &mut values,
        );
        // No core, so the first round broke on `all_empty` — and `postProcessing` still ran.
        assert_eq!(liveness.cleared, vec![VirtualAssigns::Kept]);
        assert_eq!(liveness.recomputed, vec![1]);
        assert_eq!(unit.body.len(), 1);
    }

    /// A model and a rung, so the program is typed; nothing this pass does reads either.
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

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct AnyRung;
    impl Workload for AnyRung {
        const ROWS: u32 = 1;
        const ACTIVE_CAP: u32 = 64;
    }

    /// e644 — a unit with no uniform region earns its scoreboard from the one unit value it shares,
    /// and the `lbr` copy the scoreboard REQUIRES lands in the program header: the flag is set on the
    /// op and the promotion reaches liveness, which is the whole effect of the else arm.
    #[test]
    fn e644_moves_the_required_candidate_into_the_program_header() {
        let mut program: Program<Dd2, AnyModel, AnyRung> = Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            preamble: vec![Op::Dataflow(dataflow::Op::GetUnit {
                result: Val(1),
                residency: Residency::Global,
                unit: DfirUnit::L3lu,
                num_folds: None,
                reg_locale: None,
            })],
            units: ProgramUnits::of(
                ProgramUnit::<Dd2> {
                    on: Units::one(DfirUnit::L3lu, Val(1)),
                    precision: None,
                    body: vec![
                        Op::Sentient(sentient::Op::ScalarConstant {
                            value: 4,
                            result: Val(9),
                            reg_locale: sentient::RegType::Imm,
                            ty: ScalarTy::Index,
                            is_symbol: false,
                        }),
                        copy(9, 10, None, false),
                    ],
                    arch: core::marker::PhantomData,
                },
                Vec::new(),
            ),
            bound: core::marker::PhantomData,
        };
        let mut liveness = Recording::default();
        let mut uga = OneUnit(Val(1));

        run_on_operation::<Dd2, _, _, _, _, OutOfScopeRegisterGraphs, _>(
            &mut program,
            &[Val(0)],
            &mut OutOfScopeInstructionEstimator,
            &mut liveness,
            &mut uga,
            &mut Values::default(),
        );

        let body = &program.units.iter().next().expect("one unit").body;
        assert_eq!(body[1], copy(9, 10, None, true), "the `lbr` copy is in the header");
        assert_eq!(liveness.promoted, vec![Val(10)]);
        // Nothing coalesces here, and both spendings still happen once for the one scoreboard.
        assert_eq!(liveness.optional, vec![Vec::<Vec<Val>>::new()]);
        assert_eq!(liveness.enforced, vec![Vec::<(Val, Val)>::new()]);
    }
}
