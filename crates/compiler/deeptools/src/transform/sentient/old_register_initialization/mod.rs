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

// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET, so everything below is reachable only from this
// file's own tests until `e644_runOnOperation` lands and something calls it. CI runs clippy with
// `-D warnings`, so without this the first ported leaf of the module fails the gate.
// ⭐ REMOVE THIS WITH `e644_runOnOperation`: at that point an unused item here is a real defect again.
#![allow(dead_code)]

pub(crate) mod register_init_candidate_promoter;
pub(crate) mod register_init_info;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use crate::formats::Bits;
use crate::islands::sentient::dialects::{self, Op, Val, sentient};
use crate::islands::sentient::print;

/// `ssa_weight_`'s value — *"weight is based on number of times it is accessed"* (`:122-123`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Weight(pub i32);

/// Replaces: e102_dumpWeights
///
/// Every weighted value and its weight, one `--t> N` line each.
///
/// ⛔ TRAP: `llvm::errs()` IS A DUMP AND A DUMP IS A STRING HERE — the caller (`e331_dumpWeight`)
/// wraps these lines in its own banner, so returning them is what lets it.
/// ⭐ ORDERED, WHERE `DenseMap` IS NOT: the reference's iteration order is unspecified, so a
/// `BTreeMap` is the only version of this whose output is the same twice.
/// ⚠️ A VALUE WITH NO DEFINING OP IN `scope` GETS `Value::print`'S BLOCK-ARGUMENT LINE WITHOUT ITS
/// TYPE — this island carries no per-value type. And [`print::emit`] ends its line, so `--t>` starts
/// the next one rather than following on the same one.
#[must_use]
pub fn dump_weights(ssa_weight: &BTreeMap<Val, Weight>, scope: &[Op]) -> String {
    let mut out = String::new();
    for (val, weight) in ssa_weight {
        match dialects::defining_op(*val, scope) {
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
/// ⛔ TRAP: `op->erase()` ABORTS MLIR WHILE A USE REMAINS, which is why every insertion into
/// `to_be_erased_` is preceded by a `DT_CHECK(..getUses().empty())` (`:1180`) — the rewiring is the
/// caller's, and this only carries it out.
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

// crustify:todo: e326_replaceVirtualAssignTarget
//   authority : dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:159  (8 body lines, level 1)
//   original  : void replaceVirtualAssignTarget(mlir::Value old_v, mlir::Value new_v)
//   calls     : e252_size

// crustify:todo: e331_dumpWeight
//   authority : dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:706  (5 body lines, level 1)
//   original  : void OldRegisterInitializationPass::dumpWeight( std::vector<RegisterInitInfo> &rtis)
//   calls     : e102_dumpWeights

// crustify:todo: e332_moveSSAToInit
//   authority : dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:715  (25 body lines, level 1)
//   original  : static void moveSSAToInit(mlir::Value val)
//   calls     : e252_size

// crustify:todo: e570_getFirstSource
//   authority : dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:788  (5 body lines, level 4)
//   original  : mlir::Value getFirstSource(mlir::Value core, mlir::Value val, OpBuilder &const_builder)
//   calls     : e517_getSource

// crustify:todo: e631_promoteRegisterInitCandidatesAboveUniformRegion
//   authority : dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:1036  (5 body lines, level 6)
//   original  : void OldRegisterInitializationPass:: promoteRegisterInitCandidatesAboveUniformRegion( dataflow::ProgramUnitOp &unit, std::vector<mlir::Value> &base_fold_units, Liveness &liveness, llvm::DenseMap<mlir::Value, RegisterInitInfo> &rtis, const UniformGroupAnalyzer &uga)
//   calls     : e606_run

// crustify:todo: e644_runOnOperation
//   authority : dcc/src/Transform/Sentient/OldRegisterInitialization.cpp:1203  (111 body lines, level 7)
//   original  : void OldRegisterInitializationPass::runOnOperation()
//   calls     : e107_hasUniformizeRegion, e108_removeInitAttrFromOps, e327_calcSSAWeight, e328_sortRegCoalescingCandidates, e329_collectAllRegCoalescingCandidates, e332_moveSSAToInit, e422_insert, e449_calcSSAWeight, e450_collectAllRegCoalescingCandidates, e451_collectRegInitAndRegCoalescingCandidateFast, e631_promoteRegisterInitCandidatesAboveUniformRegion

#[cfg(test)]
mod unit_tests {
    use super::{
        Weight, dump_weights, erase_deleted_ops, has_same_attr, remove_init_attr_from_ops,
    };
    use crate::formats::Bits;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::{Op, Val, sentient};
    use std::collections::{BTreeMap, BTreeSet};

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
        let mut weights = BTreeMap::new();
        weights.insert(Val(2), Weight(7));
        weights.insert(Val(9), Weight(-1));
        assert_eq!(
            dump_weights(&weights, &scope),
            "%2 = sentient.scalar_copy %1  {reg_locale = \"lbr\"} : index\n--t> 7\n<block argument> Val(9)\n--t> -1\n"
        );
    }

    /// e108 — a copy and a loop position both lose the flag, at any depth.
    #[test]
    fn e108_clears_the_program_header_flag_everywhere() {
        let mut body = vec![Op::Sentient(sentient::Op::For {
            iv: Val(3),
            bound: Val(4),
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
}
