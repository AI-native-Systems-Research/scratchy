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

//! `OpRerolling.cpp` — 8 of the campaign's 656 units (dependency level(s) [0, 1, 2, 3, 4, 5]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e122_incrementUnrollSize` | 122 | 0 | 1 | `dcc/src/Transform/Sentient/OpRerolling.hpp:131` |
//! | `e334_mergeScalarOpIntoMac` | 334 | 1 | 57 | `dcc/src/Transform/Sentient/OpRerolling.cpp:81` |
//! | `e338_setUnrollFieldsInStmt` | 338 | 1 | 311 | `dcc/src/Transform/Sentient/OpRerolling.cpp:1038` |
//! | `e453_updateRefOpUnrollInfo` | 453 | 2 | 36 | `dcc/src/Transform/Sentient/OpRerolling.cpp:139` |
//! | `e454_updateRefOpUnrollFields` | 454 | 2 | 14 | `dcc/src/Transform/Sentient/OpRerolling.cpp:181` |
//! | `e519_processOneBlock` | 519 | 3 | 239 | `dcc/src/Transform/Sentient/OpRerolling.cpp:199` |
//! | `e571_runOpRerolling` | 571 | 4 | 29 | `dcc/src/Transform/Sentient/OpRerolling.cpp:1008` |
//! | `e607_runOnOperation` | 607 | 5 | 13 | `dcc/src/Transform/Sentient/OpRerolling.cpp:994` |

// ⛔ FIVE OF THE SEVEN ANCHORS BELOW ARE UNFILLED, so nothing in the crate yet calls
// `merge_scalar_op_into_mac` or `set_unroll_fields_in_stmt` — `e519_processOneBlock` and
// `e571_runOpRerolling` are their only callers; CI runs clippy with `-D warnings`
// (the `port_assignment/mod.rs:94` precedent).
// ⭐ REMOVE THIS WITH `e607_runOnOperation`.
#![allow(dead_code)]

pub(crate) mod unroll_operands;

use core::num::NonZeroU32;

use super::utils::{SenTarget, round_down_unroll_factor};
use crate::arch::Elements;
use crate::bridges::dataflow_ir_to_sentient::vc_lowering_xrf::MacXrfIncrements;
use crate::islands::dataflow_ir::Values;
use crate::islands::sentient::dialects::sentient as sen;
use crate::islands::sentient::dialects::{
    Op, Val, defining_op, erase_defining_op, operands, regions_mut, regions_ref,
    replace_all_uses_with, results, use_count,
};
use crate::units::DfirUnit;
use unroll_operands::{
    ComputePortId, OperandName, UnrollOperands, UnrollSize as SnapshotSize,
};

/// HOW MANY OPS ONE REROLLED STATEMENT STANDS FOR — `UnrollOperands::unroll_size_`, whose declaration
/// initialises it to ONE and not zero, so an unrerolled statement already stands for itself
/// (`dcc/src/Transform/Sentient/OpRerolling.hpp:45`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct UnrollSize(pub u32);

impl Default for UnrollSize {
    fn default() -> UnrollSize {
        UnrollSize(1)
    }
}

impl UnrollSize {
    /// Replaces: e122_incrementUnrollSize
    ///
    /// Adds `incr_val` further ops to the count this rerolled statement stands for
    /// (`dcc/src/Transform/Sentient/OpRerolling.hpp:131`).
    pub fn increment(&mut self, incr_val: UnrollSize) {
        // `unroll_size_ += incr_val;` — `unsigned`, so the reference wraps rather than trapping.
        self.0 = self.0.wrapping_add(incr_val.0);
    }
}

/// ONE MAC'S SCALAR-ADD CHAIN, PLANNED BEFORE ANYTHING IS WRITTEN — see
/// [`merge_scalar_op_into_mac`], whose walk needs the block immutably and whose write needs it
/// mutably.
#[derive(Debug)]
struct XrfMerge {
    /// `xrf_type == xrf_write`.
    write: bool,
    /// `accum_xrf_ptr_incr_value` once the chain has been followed to its end.
    accum: i32,
    /// The value the last add in the chain binds — `result` where the loop stopped.
    tail: Val,
    /// `mac_op.getResult(write ? 0 : 1)`, which the chain's readers are re-pointed at.
    mac_result: Val,
    /// `tmp` — the adds and their sole-use constants, in push order.
    doomed: Vec<Val>,
}

/// Replaces: e334_mergeScalarOpIntoMac
///
/// Folds every xrf-pointer `sentient.scalar_add` chain hanging off a MAC's pointer result into that
/// MAC's own `xrfReadIncr`/`xrfWriteIncr`, then erases the adds and their constants (`:81`).
///
/// ⛔ TWO PHASES BECAUSE `hasOneUse()` ASKS ABOUT THE WHOLE BLOCK: every use question is answered
/// against `bb` before one attribute is written, and the erases run LAST in REVERSE push order — the
/// reference's `pop_back` — so an add goes before the constants it consumed (`:130-136`).
pub(crate) fn merge_scalar_op_into_mac(bb: &mut Vec<Op>) {
    let mut merges = Vec::new();
    let block: &[Op] = bb;
    // `bb->walk([&](sentient::MacOp mac_op) { .. })`.
    collect_xrf_merges(block, block, &mut merges);
    // `mac_op.setXrfReadIncrAttr(..)` / `setXrfWriteIncrAttr(..)` (`:117-124`).
    apply_xrf_merges(bb, &merges);
    let mut doomed = Vec::new();
    for merge in &merges {
        // `result.replaceAllUsesWith(mac_op.getResult(..))` (`:120`, `:124`).
        replace_all_uses_with(bb, merge.tail, merge.mac_result);
        // `scalar_op_to_be_deleted.append(tmp);` — ONE list across every merged mac (`:126`).
        doomed.extend(merge.doomed.iter().copied());
    }
    // `while (!scalar_op_to_be_deleted.empty()) { back(); pop_back(); dropAllUses(); erase(); }`.
    while let Some(val) = doomed.pop() {
        erase_defining_op(bb, val);
    }
}

/// The plan half of [`merge_scalar_op_into_mac`] — a post-order walk that reaches nested regions and
/// asks every `hasOneUse()` of `root` (`:92-129`).
fn collect_xrf_merges(scope: &[Op], root: &[Op], out: &mut Vec<XrfMerge>) {
    for op in scope {
        for region in regions_ref(op) {
            collect_xrf_merges(region, root, out);
        }
        let Op::Sentient(sen_op) = op else { continue };
        let sen::Op::VectorMac {
            results,
            xrf_read_incr,
            xrf_write_incr,
            ..
        } = sen_op
        else {
            continue;
        };
        // `if (mac_op.isXrfRelated())`, then `isXrfWtRelated() ? xrf_write : xrf_read` (`:93-94`).
        // ⭐ BOTH PREDICATES ALREADY LIVE ON `MacXrfIncrements`, which needs the op mutably to hand
        // back its increment slots, so the probe is a throwaway copy rather than a second port.
        let mut probe = sen_op.clone();
        let Some(related) = MacXrfIncrements::of(&mut probe) else {
            continue;
        };
        if !(related.rd_related() || related.wt_related()) {
            continue;
        }
        let write = related.wt_related();
        // `getXrfWriteIncrSigned()` / `getXrfReadIncrSigned()` (`:96-98`).
        let mut accum = if write {
            *xrf_write_incr as i32
        } else {
            *xrf_read_incr as i32
        };
        // `mac_op.getResult(0)` for the write pointer, `getResult(1)` for the read one (`:100`).
        let index = usize::from(!write);
        let Some(&mac_result) = results.get(index) else {
            panic!(
                "mac_op.getResult({index}) on a mac binding {} results (`OpRerolling.cpp:100`)",
                results.len()
            )
        };
        let mut tail = mac_result;
        let mut doomed = Vec::new();
        // `while (result.hasOneUse() && isa<sentient::AddOp>(*result.getUsers().begin()))` (`:101`).
        while use_count(tail, root) == 1
            && let Some(Op::Sentient(sen::Op::ScalarAdd { lhs, rhs, result, .. })) =
                sole_user(tail, root)
        {
            accum += add_increment_val(*lhs, *rhs, root) as i32;
            tail = *result;
            for operand in [*lhs, *rhs] {
                // `isa<sentient::ConstantOp>(parent_op) && parent_op->hasOneUse()` (`:107`).
                // ⛔ A NULL `parent_op` IS THE REFERENCE'S OWN `isa` ASSERT; answered `false` here.
                if matches!(
                    defining_op(operand, root),
                    Some(Op::Sentient(sen::Op::ScalarConstant { .. }))
                ) && use_count(operand, root) == 1
                {
                    doomed.push(operand);
                }
            }
            doomed.push(*result);
        }
        // `if (result.getDefiningOp() != mac_op)` — nothing moved unless the loop ran (`:115`).
        if tail != mac_result {
            out.push(XrfMerge {
                write,
                accum,
                tail,
                mac_result,
                doomed,
            });
        }
    }
}

/// `*result.getUsers().begin()` — asked only where [`use_count`] has already said there is exactly one
/// use, so "the first" and "the only" name the same op (`:102-103`).
fn sole_user(of: Val, scope: &[Op]) -> Option<&Op> {
    for op in scope {
        if operands(op).contains(&of) {
            return Some(op);
        }
        for region in regions_ref(op) {
            if let Some(found) = sole_user(of, region) {
                return Some(found);
            }
        }
    }
    None
}

/// `getIncrementVal(add_op)`'s `AddOp` arm — whichever of the two operands is a constant
/// (`SentientOps.cpp:2085-2100`).
///
/// ⛔ BOTH REFUSALS ARE THE REFERENCE'S OWN: two constant operands trips its `DT_CHECK` (`:2088`) and
/// neither being constant falls through to `DT_ERROR("operation does not have an increment value!")`.
fn add_increment_val(lhs: Val, rhs: Val, scope: &[Op]) -> i64 {
    let value_of = |val: Val| match defining_op(val, scope) {
        Some(Op::Sentient(sen::Op::ScalarConstant { value, .. })) => Some(*value),
        _ => None,
    };
    match (value_of(lhs), value_of(rhs)) {
        (Some(_), Some(_)) => panic!(
            "DT_CHECK(!isa_and_nonnull<sentient::ConstantOp>(..)) \
             (`SentientOps.cpp:2088`): both operands of one scalar_add are constants"
        ),
        (_, Some(value)) | (Some(value), None) => value,
        (None, None) => panic!(
            "DT_ERROR(\"operation does not have an increment value!\") (`SentientOps.cpp:2099`)"
        ),
    }
}

/// The write half of [`merge_scalar_op_into_mac`] — each planned mac, found by a result it binds
/// (`:117-124`).
fn apply_xrf_merges(scope: &mut [Op], merges: &[XrfMerge]) {
    for op in scope.iter_mut() {
        if let Op::Sentient(sen::Op::VectorMac {
            results,
            xrf_read_incr,
            xrf_write_incr,
            ..
        }) = op
            && let Some(merge) = merges
                .iter()
                .find(|merge| results.contains(&merge.mac_result))
        {
            if merge.write {
                *xrf_write_incr = merge.accum as u32;
            } else {
                *xrf_read_incr = merge.accum as u32;
            }
        }
        for region in regions_mut(op) {
            apply_xrf_merges(region, merges);
        }
    }
}

/// WHETHER OP REROLLING FINISHED WITH A STATEMENT — `signalPassFailure()` is STICKY AND DOES NOT
/// BREAK the clone loop, so an unsupported op fails the pass without stopping the rewrite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Rerolled {
    /// Every clone the leftover unroll size asked for was created.
    Done,
    /// `op->emitError("This operation is not supported by opRerolling."); signalPassFailure();`
    /// (`OpRerolling.cpp:1347-1349`).
    Unsupported,
}

/// WHICH DECLARED OPERAND OF A COMPUTE an attribute name splices in — the `"A"`/`"B"`/`"C"` the
/// reference builds `"unrollIncrOp" + name`, `"op" + name + "PortID"` and `"..DataID"` from
/// (`:1123-1145`).
///
/// ⛔ NOT [`OperandName`], WHICH IS A PORT SLOT. This names a FIELD of the op; the two are keyed
/// against each other by `port_to_op_conversion`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpSlot {
    /// `opA`.
    A,
    /// `opB`.
    B,
    /// `opC`.
    C,
}

/// Replaces: e338_setUnrollFieldsInStmt
///
/// Writes the rerolled unroll factor, per-operand increment flags, reset port and data ids and xrf
/// increments onto `body[at]`, then CLONES it until the snapshot's unroll size is spent (`:1038`).
///
/// ⛔ THE CLONES ARE THE PORT, not a tidy-up: each takes one rounded factor's worth, and a mac or a
/// transfer hands its own results to the next clone so the pointer/address chain stays wired
/// (`:1226-1244`, `:1314-1330`). ⭐ `at` is the reference's `Operation *op`, WHICH IT REASSIGNS.
pub(crate) fn set_unroll_fields_in_stmt(
    body: &mut Vec<Op>,
    at: usize,
    operand_list: &UnrollOperands,
    component: DfirUnit,
    sen_target: SenTarget,
    values: &mut Values,
) -> Rerolled {
    // `if (operand_list.getOpName() == "NA" || !operand_list.areUnrollFieldsUpdated()) return;`
    if operand_list.op_name.is_none() || !operand_list.are_unroll_fields_updated {
        return Rerolled::Done;
    }
    let mut unroll_size = operand_list.unroll_size.0;
    let mut remaining_xrf_write_incr = operand_list.xrf_write_incr.0;
    let is_memory_unit = operand_list.is_memory_unit;
    // `unroll_factor_val = round_down(unroll_size); unroll_size -= unroll_factor_val;` (`:1061-1064`).
    let mut unroll_factor_val = round_down(unroll_size, component, &body[at], sen_target);
    unroll_size -= unroll_factor_val;
    let result_unrolled = operand_list.is_field_unroll[OperandName::Result.slot()];

    // `if (auto splat_op = llvm::dyn_cast<sentient::SplatOp>(op))` — first, and it returns (`:1071`).
    if matches!(body[at], Op::Sentient(sen::Op::Splat { .. })) {
        if let Op::Sentient(sen::Op::Splat {
            unroll_incr_result,
            unroll_factor,
            ..
        }) = &mut body[at]
        {
            *unroll_incr_result = result_unrolled;
            *unroll_factor = unroll_factor_of(unroll_factor_val);
        }
        // `builder.setInsertionPointAfter(op)`, which MLIR advances past each insert (`:1069`).
        let mut insert_at = at + 1;
        while unroll_size > 0 {
            // `builder.clone(*operand_list.getSplatInput().getDefiningOp())` (`:1085-1086`).
            let Some(splat_input) = operand_list.splat_input else {
                panic!(
                    "getSplatInput().getDefiningOp() on an unfilled splat input \
                     (`OpRerolling.cpp:1085`)"
                )
            };
            let Some(input_def) = defining_op(splat_input, body) else {
                panic!("getSplatInput().getDefiningOp() is null (`OpRerolling.cpp:1085`)")
            };
            let mut new_input = input_def.clone();
            let Some(&new_input_result) = remint_results(&mut new_input, values).first() else {
                panic!("new_input_op->getResult(0) on an op binding nothing (`OpRerolling.cpp:1110`)")
            };
            // `createOperand(result, unroll_size)` then a fresh `sentient.logical_port` (`:1087-1092`).
            let Some(new_output_port) =
                operand_list.create_operand(OperandName::Result, SnapshotSize(unroll_size))
            else {
                panic!(
                    "symbolizeSentientComputePort(new_output_port).value() on std::nullopt \
                     (`OpRerolling.cpp:1091`)"
                )
            };
            let new_output_result = values.mint();
            let new_output = Op::Sentient(sen::Op::LogicalPort {
                port_name: new_output_port,
                result: new_output_result,
            });
            unroll_factor_val = round_down(unroll_size, component, &body[at], sen_target);
            unroll_size -= unroll_factor_val;
            // `builder.clone(*op)` with the new factor, `unrollIncrResult = (unroll_factor_val > 1)`
            // and its two operands re-pointed at the pair just made (`:1101-1111`).
            let mut new_op = body[at].clone();
            if let Op::Sentient(sen::Op::Splat {
                input,
                output,
                unroll_factor,
                unroll_incr_result,
                ..
            }) = &mut new_op
            {
                *unroll_factor = unroll_factor_of(unroll_factor_val);
                *unroll_incr_result = unroll_factor_val > 1;
                *input = new_input_result;
                *output = new_output_result;
            }
            for made in [new_input, new_output, new_op] {
                body.insert(insert_at, made);
                insert_at += 1;
            }
        }
        return Rerolled::Done;
    }

    let op_a_unrolled = operand_list.is_field_unroll[OperandName::OpA.slot()];
    let op_b_unrolled = operand_list.is_field_unroll[OperandName::OpB.slot()];
    let op_c_unrolled = operand_list.is_field_unroll[OperandName::OpC.slot()];

    if is_memory_unit {
        // `op->setAttr("burst_size", unroll_factor_val)` — all a transfer takes (`:1148`).
        set_burst_size(&mut body[at], unroll_factor_val);
    } else {
        let Op::Sentient(sen_op) = &mut body[at] else {
            panic!("setUnrollFieldsInStmt over an op of another dialect (`OpRerolling.cpp:1150`)")
        };
        set_unroll_factor(sen_op, unroll_factor_of(unroll_factor_val));
        // `op->setAttr("unrollIncrResult", result_unroll_attr)` (`:1156`).
        if let Some(result) = result_ports_mut(sen_op) {
            result.unroll_incr = result_unrolled;
        }
        // `auto op_a_name = port_to_op_conversion(0, op);` (`:1157-1158`).
        let op_a_slot = port_to_op_conversion(0, sen_op);
        if let Some(slot) = op_a_slot
            && let Some(operand) = operand_mut(sen_op, slot)
        {
            operand.unroll_incr = op_a_unrolled;
        }
        // Which of the three op_b arms applies (`:1160-1189`): `is_any_of(type, PE, SFP)` and, inside
        // that, whether the op is a binary and whether its operator is a multiply.
        let binary_mul = if let sen::Op::VectorBinary { binary_op, .. } = &*sen_op {
            Some(matches!(
                binary_op.op(),
                sen::BinaryOp::Mul | sen::BinaryOp::MulDiv2
            ))
        } else {
            None
        };
        let pe_or_sfp = matches!(component, DfirUnit::Pe | DfirUnit::Sfp);
        if pe_or_sfp && binary_mul.is_some() {
            // `op->setAttr("unrollIncrLogicalResult", ..)` (`:1162-1165`).
            // ⛔ THE ISLAND CARRIES THIS FLAG ON THE FORWARD ITSELF, so a `Binary::Plain` has nowhere
            // to hold it and needs none: e337 skips an empty logical-result forwarding list (`:912`)
            // and e336 compares it equal, so nothing downstream can read it.
            let logical_unrolled = operand_list.is_field_unroll[OperandName::LogicalResult.slot()];
            if let sen::Op::VectorBinary {
                binary_op: sen::Binary::Forwarding { unroll_incr, .. },
                ..
            } = sen_op
            {
                *unroll_incr = logical_unrolled;
            }
        }
        let (op_b_port, op_b_unroll_incr) = match (pe_or_sfp, binary_mul) {
            (true, Some(true) | None) => (1, op_b_unrolled),
            // ⭐ AND THE REFERENCE WRITES `op_c_unroll_attr` HERE — its own `// why this diff?` (`:1180`).
            (true, Some(false)) => (2, op_c_unrolled),
            (false, _) => (2, op_b_unrolled),
        };
        let op_b_slot = port_to_op_conversion(op_b_port, sen_op);
        if let Some(slot) = op_b_slot
            && let Some(operand) = operand_mut(sen_op, slot)
        {
            operand.port_id = Some(op_b_port);
            operand.unroll_incr = op_b_unroll_incr;
        }
        // `Reset dataId's` (`:1191-1193`) — ⛔ `None` IS THE `.td`'S `-1`, not data id zero.
        for slot in [op_a_slot, op_b_slot].into_iter().flatten() {
            if let Some(operand) = operand_mut(sen_op, slot) {
                operand.data_id = None;
            }
        }
        // `Reset portID's` (`:1195-1196`).
        if let Some(slot) = op_a_slot
            && let Some(operand) = operand_mut(sen_op, slot)
        {
            operand.port_id = Some(0);
        }
        if matches!(sen_op, sen::Op::VectorMac { .. }) {
            // ⛔ READ AFTER THE RESETS ABOVE, which is what makes this a THIRD lookup and not a copy
            // of `op_b_name`: the port ids it searches have just been rewritten (`:1199`).
            let op_c_port = if component.is_pt_row() { 1 } else { 2 };
            let op_c_slot = port_to_op_conversion(op_c_port, sen_op);
            if let Some(slot) = op_c_slot
                && let Some(operand) = operand_mut(sen_op, slot)
            {
                operand.unroll_incr = op_c_unrolled;
                operand.data_id = None;
                operand.port_id = Some(op_c_port);
            }
            // Only the LAST op of the rerolled chain carries a real increment (`:1204-1216`).
            if operand_list.is_xrf_rd_op() {
                let value = if unroll_size == 0 {
                    operand_list.xrf_read_incr.0
                } else {
                    0
                };
                if let sen::Op::VectorMac { xrf_read_incr, .. } = sen_op {
                    *xrf_read_incr = value as u32;
                }
            } else if operand_list.is_xrf_wt_op() {
                let value = if unroll_size == 0 {
                    remaining_xrf_write_incr
                } else {
                    unroll_factor_val as i32
                };
                remaining_xrf_write_incr -= value;
                if let sen::Op::VectorMac { xrf_write_incr, .. } = sen_op {
                    *xrf_write_incr = value as u32;
                }
            }
        }
    }

    // `while (unroll_size > 0)` — one clone per rounded factor (`:1224`).
    let mut cur = at;
    let mut insert_at = at + 1;
    while unroll_size > 0 {
        if is_memory_unit {
            // `builder.clone(*op)`, the new burst size, `op->replaceAllUsesWith(new_op)` and only THEN
            // `new_op->setOperand(mutable_addr, op->getResult(0))` — that order is load-bearing, or the
            // clone's own address operand would be rewritten to itself (`:1226-1244`).
            let mut new_op = body[cur].clone();
            unroll_factor_val = round_down(unroll_size, component, &body[cur], sen_target);
            unroll_size -= unroll_factor_val;
            set_burst_size(&mut new_op, unroll_factor_val);
            let old_results = results(&body[cur]);
            let fresh = remint_results(&mut new_op, values);
            for (old, new) in old_results.iter().zip(&fresh) {
                replace_all_uses_with(body, *old, *new);
            }
            let Some(&old_result) = old_results.first() else {
                panic!("op->getResult(0) on a transfer binding nothing (`OpRerolling.cpp:1242`)")
            };
            set_mutable_addr(&mut new_op, old_result);
            body.insert(insert_at, new_op);
            // `op = new_op;`
            cur = insert_at;
            insert_at += 1;
            continue;
        }

        let Op::Sentient(cur_op) = &body[cur] else {
            panic!("op rerolling over an op of another dialect (`OpRerolling.cpp:1249`)")
        };
        // `getUnrollOperandUsingPort(type, op<X>PortID)` → `createOperand` + `createForwardingArray`,
        // for A always and for B and C only where the op declares them (`:1249-1281`).
        let (operand_a, op_a_forwarding) =
            rerolled_slot(operand_list, component, cur_op, OpSlot::A, unroll_size);
        let (operand_b, op_b_forwarding) =
            rerolled_slot(operand_list, component, cur_op, OpSlot::B, unroll_size);
        let (operand_c, op_c_forwarding) =
            rerolled_slot(operand_list, component, cur_op, OpSlot::C, unroll_size);
        let result_forwarding =
            operand_list.create_forwarding_array(OperandName::Result, SnapshotSize(unroll_size));
        unroll_factor_val = round_down(unroll_size, component, &body[cur], sen_target);
        unroll_size -= unroll_factor_val;
        let mut new_op = body[cur].clone();
        let fresh = remint_results(&mut new_op, values);
        // ⭐ BOTH INCREMENTS ARE COMPUTED FROM THE ALREADY-DECREMENTED SIZE (`:1294-1304`).
        let xrf_read_incr_val = if operand_list.is_xrf_rd_op() && unroll_size == 0 {
            operand_list.xrf_read_incr.0
        } else {
            0
        };
        let xrf_write_incr_val = if operand_list.is_xrf_wt_op() {
            if unroll_size == 0 {
                remaining_xrf_write_incr
            } else {
                unroll_factor_val as i32
            }
        } else {
            0
        };
        remaining_xrf_write_incr -= xrf_write_incr_val;
        let old_results = results(&body[cur]);
        let Op::Sentient(new_sen) = &mut new_op else {
            panic!("builder.clone(*op) of an op of another dialect (`OpRerolling.cpp:1285`)")
        };
        set_unroll_factor(new_sen, unroll_factor_of(unroll_factor_val));
        let Some(result_forwarding) = result_forwarding else {
            panic!(
                "symbolizeSentientComputePort on a ResultForwarding entry (`OpRerolling.cpp:952`)"
            )
        };
        if let Some(result) = result_ports_mut(new_sen) {
            result.forwarding = result_forwarding;
        }
        let mut advance = false;
        if matches!(new_sen, sen::Op::VectorMac { .. }) {
            set_slot(new_sen, OpSlot::A, operand_a, op_a_forwarding);
            set_slot(new_sen, OpSlot::B, operand_b, op_b_forwarding);
            set_slot(new_sen, OpSlot::C, operand_c, op_c_forwarding);
            // `if (operand_list.isXrfOp())` — the pointers, then the two increments (`:1314-1330`).
            if operand_list.is_xrf_op() {
                if let sen::Op::VectorMac {
                    xrf_write_ptr,
                    xrf_read_ptr,
                    xrf_read_incr,
                    xrf_write_incr,
                    ..
                } = new_sen
                {
                    // `for (auto &pointer_mutable : new_mac_op.getPointersMutable())` — ⛔ WRITE
                    // POINTER FIRST, and the ordinal counts only the pointers the op HAS.
                    let mut bound = 0;
                    for pointer in [xrf_write_ptr, xrf_read_ptr] {
                        if pointer.is_some() {
                            let Some(&result) = old_results.get(bound) else {
                                panic!(
                                    "op->getResult({bound}) on a mac binding {} results \
                                     (`OpRerolling.cpp:1318`)",
                                    old_results.len()
                                )
                            };
                            *pointer = Some(result);
                            bound += 1;
                        }
                    }
                    *xrf_read_incr = xrf_read_incr_val as u32;
                    *xrf_write_incr = xrf_write_incr_val as u32;
                }
                // `replaceAllUsesExcept(new_mac_op->getResult(i), {new_mac_op})` (`:1324-1327`) — the
                // exception is free here because the clone is not in the body yet.
                for (old, new) in old_results.iter().zip(&fresh) {
                    replace_all_uses_with(body, *old, *new);
                }
                advance = true;
            }
        } else if matches!(new_sen, sen::Op::VectorBinary { .. }) {
            set_slot(new_sen, OpSlot::A, operand_a, op_a_forwarding);
            set_slot(new_sen, OpSlot::B, operand_b, op_b_forwarding);
        } else if matches!(new_sen, sen::Op::VectorUnary { .. }) {
            set_slot(new_sen, OpSlot::A, operand_a, op_a_forwarding);
        } else {
            // `op->emitError(..); signalPassFailure();` — THE PASS HAS FAILED, so the clone the
            // reference leaves behind is never read and is not created here (`:1347-1349`).
            return Rerolled::Unsupported;
        }
        body.insert(insert_at, new_op);
        // `op = new_mac_op;` happens for an xrf mac alone — a binary, a unary and a non-xrf mac all
        // leave the ORIGINAL as the thing the next clone is made from (`:1330`).
        if advance {
            cur = insert_at;
        }
        insert_at += 1;
    }
    Rerolled::Done
}

/// `round_down` (`:1048-1059`) — how many ops the next clone stands for.
///
/// ⛔ THE REFERENCE'S `type == L3` ARM IS UNREACHABLE: `runOpRerolling`'s gate admits SFP, PE, PT, LX,
/// L0, LXLU, LXSU, L0LU and L0SU and NOT L3 (`:1017`), and there is no plain `L3` [`DfirUnit`].
fn round_down(n: u32, component: DfirUnit, op: &Op, sen_target: SenTarget) -> u32 {
    match component {
        DfirUnit::Lx
        | DfirUnit::L0
        | DfirUnit::Lxlu
        | DfirUnit::Lxsu
        | DfirUnit::L0lu
        | DfirUnit::L0su => n.min(64),
        _ => match NonZeroU32::new(n) {
            Some(n) => round_down_unroll_factor(n, op, sen_target).count(),
            None => panic!(
                "DT_CHECK_MSG(n != 0, \"Expect a positive target unroll value\") \
                 (`Transform/Sentient/Utils.cpp:399`)"
            ),
        },
    }
}

/// `symbolizeSentientUnrollFactor("x" + std::to_string(n)).value()` (`:1069`, `:1104`, `:1288`).
///
/// ⛔ PAST THE FIVE THE DIALECT NAMES THIS IS THE REFERENCE'S OWN `.value()` ON A `std::nullopt`: the
/// 64 and 32 [`round_down`] can return are burst sizes, and only the memory path avoids coming here.
fn unroll_factor_of(n: u32) -> sen::UnrollFactor {
    match n {
        1 => sen::UnrollFactor::X1,
        2 => sen::UnrollFactor::X2,
        3 => sen::UnrollFactor::X3,
        4 => sen::UnrollFactor::X4,
        8 => sen::UnrollFactor::X8,
        _ => panic!(
            "symbolizeSentientUnrollFactor(\"x{n}\").value() on std::nullopt \
             (`OpRerolling.cpp:1069`)"
        ),
    }
}

/// `port_to_op_conversion` (`:1123-1145`) — which FIELD of this op carries the operand assigned to
/// `port_id`, mirroring the reference's `dyn_cast` chain.
///
/// ⛔ `None` IS THE REFERENCE'S EMPTY NAME, whose `setAttr("unrollIncrOp" + "")` writes an attribute
/// nothing reads — a unary asked for port1 or port2 is exactly that case (`:1140-1143`).
fn port_to_op_conversion(port_id: i32, op: &sen::Op) -> Option<OpSlot> {
    if let sen::Op::VectorMac { op_a, op_b, .. } = op {
        Some(if op_a.port_id == Some(port_id) {
            OpSlot::A
        } else if op_b.port_id == Some(port_id) {
            OpSlot::B
        } else {
            OpSlot::C
        })
    } else if let sen::Op::VectorBinary { op_a, .. } = op {
        Some(if op_a.port_id == Some(port_id) {
            OpSlot::A
        } else {
            OpSlot::B
        })
    } else if let sen::Op::VectorUnary { op_a, .. } = op {
        (op_a.port_id == Some(port_id)).then_some(OpSlot::A)
    } else {
        None
    }
}

/// One declared operand of a compute — `None` where the op has none, which is the reference's
/// `op->hasAttr("opBPortID")` / `"opCPortID"` test (`:1265`, `:1273`).
fn operand_ref(op: &sen::Op, slot: OpSlot) -> Option<&sen::Operand> {
    if let sen::Op::VectorMac {
        op_a, op_b, op_c, ..
    } = op
    {
        Some(match slot {
            OpSlot::A => op_a,
            OpSlot::B => op_b,
            OpSlot::C => op_c,
        })
    } else if let sen::Op::VectorBinary { op_a, op_b, .. } = op {
        match slot {
            OpSlot::A => Some(op_a),
            OpSlot::B => Some(op_b),
            OpSlot::C => None,
        }
    } else if let sen::Op::VectorUnary { op_a, .. } = op {
        match slot {
            OpSlot::A => Some(op_a),
            OpSlot::B | OpSlot::C => None,
        }
    } else {
        None
    }
}

/// [`operand_ref`], mutably.
fn operand_mut(op: &mut sen::Op, slot: OpSlot) -> Option<&mut sen::Operand> {
    if let sen::Op::VectorMac {
        op_a, op_b, op_c, ..
    } = op
    {
        Some(match slot {
            OpSlot::A => op_a,
            OpSlot::B => op_b,
            OpSlot::C => op_c,
        })
    } else if let sen::Op::VectorBinary { op_a, op_b, .. } = op {
        match slot {
            OpSlot::A => Some(op_a),
            OpSlot::B => Some(op_b),
            OpSlot::C => None,
        }
    } else if let sen::Op::VectorUnary { op_a, .. } = op {
        match slot {
            OpSlot::A => Some(op_a),
            OpSlot::B | OpSlot::C => None,
        }
    } else {
        None
    }
}

/// `op->setAttr("unrollIncrResult", ..)` and `"ResultForwarding"` reach the same bundle on all three
/// computes.
fn result_ports_mut(op: &mut sen::Op) -> Option<&mut sen::ResultPorts> {
    if let sen::Op::VectorMac { result, .. }
    | sen::Op::VectorBinary { result, .. }
    | sen::Op::VectorUnary { result, .. } = op
    {
        Some(result)
    } else {
        None
    }
}

/// `op->setAttr("unrollFactor", SentientUnrollFactorAttr::get(..))` (`:1074`, `:1102`, `:1286`).
fn set_unroll_factor(op: &mut sen::Op, factor: sen::UnrollFactor) {
    if let sen::Op::VectorMac { unroll_factor, .. }
    | sen::Op::VectorBinary { unroll_factor, .. }
    | sen::Op::VectorUnary { unroll_factor, .. }
    | sen::Op::Splat { unroll_factor, .. } = op
    {
        *unroll_factor = factor;
    } else {
        panic!(
            "op->setAttr(\"unrollFactor\", ..) on an op with no unroll factor \
             (`OpRerolling.cpp:1151`)"
        )
    }
}

/// `op->setAttr("burst_size", builder.getI32IntegerAttr(unroll_factor_val))` (`:1148`, `:1231`) —
/// the transfer's own extent field, which e335 turned into the unroll size to begin with.
fn set_burst_size(op: &mut Op, burst: u32) {
    if let Op::Sentient(
        sen::Op::LoadAndSend { extent, .. } | sen::Op::ReceiveAndStore { extent, .. },
    ) = op
    {
        extent.burst_size = Elements(u64::from(burst));
    } else {
        panic!(
            "llvm_unreachable(\"Ineligible operation for op rerolling!\") \
             (`OpRerolling.cpp:1241`): {op:?}"
        )
    }
}

/// `new_op->setOperand(load_and_send_op.getMutableAddrMutable().getOperandNumber(), ..)` (`:1235-1243`).
fn set_mutable_addr(op: &mut Op, val: Val) {
    if let Op::Sentient(
        sen::Op::LoadAndSend { mutable_addr, .. } | sen::Op::ReceiveAndStore { mutable_addr, .. },
    ) = op
    {
        *mutable_addr = val;
    } else {
        panic!(
            "llvm_unreachable(\"Ineligible operation for op rerolling!\") \
             (`OpRerolling.cpp:1241`): {op:?}"
        )
    }
}

/// `setOp<X>Attr(symbolize(operand_x).value())` and `setOp<X>ForwardingAttr(..)` (`:1305-1313`).
///
/// ⛔ BOTH `None`s ARE `.value()` ON A `std::nullopt` in the reference, not checks added here — see
/// [`UnrollOperands::create_operand`].
fn set_slot(
    op: &mut sen::Op,
    slot: OpSlot,
    port: Option<sen::Port>,
    forwarding: Option<Vec<sen::Port>>,
) {
    let Some(port) = port else {
        panic!(
            "symbolizeSentientComputePort(operand_{slot:?}).value() on std::nullopt \
             (`OpRerolling.cpp:1306`)"
        )
    };
    let Some(forwarding) = forwarding else {
        panic!(
            "symbolizeSentientComputePort on an op{slot:?} forwarding entry \
             (`OpRerolling.cpp:952`)"
        )
    };
    if let Some(operand) = operand_mut(op, slot) {
        operand.port = port;
        operand.forwarding = forwarding;
    }
}

/// The operand and forwarding array one slot of a clone should carry, keyed by the port id the CURRENT
/// op holds (`:1249-1281`).
fn rerolled_slot(
    operand_list: &UnrollOperands,
    component: DfirUnit,
    op: &sen::Op,
    slot: OpSlot,
    unroll_size: u32,
) -> (Option<sen::Port>, Option<Vec<sen::Port>>) {
    let Some(operand) = operand_ref(op, slot) else {
        return (None, None);
    };
    let name = UnrollOperands::unroll_operand_using_port(
        component,
        ComputePortId::from_port_id(operand.port_id),
    );
    (
        operand_list.create_operand(name, SnapshotSize(unroll_size)),
        operand_list.create_forwarding_array(name, SnapshotSize(unroll_size)),
    )
}

/// `builder.clone(*op)` BINDS FRESH VALUES — MLIR mints them with the op, here they are minted by
/// hand and handed back in result order.
fn remint_results(op: &mut Op, values: &mut Values) -> Vec<Val> {
    let Op::Sentient(inner) = op else {
        panic!("builder.clone(*op) of an op of another dialect (`OpRerolling.cpp:1285`)")
    };
    sen::results_mut(inner)
        .into_iter()
        .map(|slot| {
            let fresh = values.mint();
            *slot = fresh;
            fresh
        })
        .collect()
}

// crustify:todo: e453_updateRefOpUnrollInfo
//   authority : dcc/src/Transform/Sentient/OpRerolling.cpp:139  (36 body lines, level 2)
//   original  : void OpRerollingPass::updateRefOpUnrollInfo( mlir::Operation *curr_op, mlir::Operation *ref_op, mlir::Operation *prev_op, mlir::Operation *last_op, sentient::UnrollOperands &ref_operand_list, sentient::UnrollOperands &curr_operand_list, llvm::SmallVector<mlir::Operation *> &ops_to_be_deleted, SenCom
//   calls     : e113_isXrfRdOp, e114_isXrfWtOp, e115_reset, e122_incrementUnrollSize, e337_updateUnrollInfo, e338_setUnrollFieldsInStmt

// crustify:todo: e454_updateRefOpUnrollFields
//   authority : dcc/src/Transform/Sentient/OpRerolling.cpp:181  (14 body lines, level 2)
//   original  : void OpRerollingPass::updateRefOpUnrollFields( mlir::Operation *curr_op, mlir::Operation *ref_op, mlir::Operation *prev_op, sentient::UnrollOperands &ref_operand_list, sentient::UnrollOperands &curr_operand_list, SenComponents type)
//   calls     : e115_reset, e338_setUnrollFieldsInStmt

// crustify:todo: e519_processOneBlock
//   authority : dcc/src/Transform/Sentient/OpRerolling.cpp:199  (239 body lines, level 3)
//   original  : void OpRerollingPass::processOneBlock(SenComponents type, Block *bb)
//   calls     : e115_reset, e248_getFoldModeAttributeIfExists, e335_fill, e336_match, e338_setUnrollFieldsInStmt, e453_updateRefOpUnrollInfo

// crustify:todo: e571_runOpRerolling
//   authority : dcc/src/Transform/Sentient/OpRerolling.cpp:1008  (29 body lines, level 4)
//   original  : void OpRerollingPass::runOpRerolling(Operation *op, bool merge_xrf_into_mac)
//   calls     : e252_size, e334_mergeScalarOpIntoMac, e519_processOneBlock

// crustify:todo: e607_runOnOperation
//   authority : dcc/src/Transform/Sentient/OpRerolling.cpp:994  (13 body lines, level 5)
//   original  : void OpRerollingPass::runOnOperation()
//   calls     : e571_runOpRerolling

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::sentient::{
        FmaMode, Operand, Precision, RegType, ResultPorts, UnaryOp, UnrollFactor,
    };
    use unroll_operands::RolledOp;

    /// AN UNREROLLED STATEMENT ALREADY STANDS FOR ONE OP, and each merge adds its own count.
    #[test]
    fn incrementing_adds_to_a_size_that_starts_at_one() {
        let mut size = UnrollSize::default();
        assert_eq!(size, UnrollSize(1));
        size.increment(UnrollSize(3));
        assert_eq!(size, UnrollSize(4));
    }

    /// A MAC WHOSE READ POINTER FEEDS A `scalar_add` CHAIN keeps the chain's total as its own
    /// increment, and the adds and their sole-use constants go.
    #[test]
    fn merging_folds_a_scalar_add_chain_into_the_macs_read_increment() {
        let mut body = vec![
            Op::Sentient(sen::Op::VectorMac {
                mask: None,
                xrf_write_ptr: None,
                xrf_read_ptr: None,
                results: vec![Val(0), Val(1)],
                op_a: Operand::from(sen::Port::Xrf),
                op_b: Operand::from(sen::Port::West),
                op_c: Operand::from(sen::Port::Zero),
                result: ResultPorts::default(),
                mode: FmaMode::FusedMulAdd,
                compute_precision: Precision::Fp16,
                fold_mode: None,
                unroll_factor: UnrollFactor::X1,
                xrf_read_incr: 2,
                xrf_write_incr: 0,
                data_transfer_only: false,
                is_data_weight: None,
                dbg_name: None,
            }),
            Op::Sentient(sen::Op::ScalarConstant {
                value: 5,
                result: Val(2),
                reg_locale: RegType::Imm,
                ty: ScalarTy::Index,
                is_symbol: false,
            }),
            Op::Sentient(sen::Op::ScalarAdd {
                lhs: Val(1),
                rhs: Val(2),
                result: Val(3),
                reg: None,
                element_size: None,
                ty: ScalarTy::Index,
            }),
            // ⭐ READS `Val(3)` TWICE, so the chain walk stops here — `hasOneUse()` is false.
            Op::Sentient(sen::Op::ScalarAdd {
                lhs: Val(3),
                rhs: Val(3),
                result: Val(4),
                reg: None,
                element_size: None,
                ty: ScalarTy::Index,
            }),
        ];
        merge_scalar_op_into_mac(&mut body);
        let Op::Sentient(sen::Op::VectorMac { xrf_read_incr, .. }) = &body[0] else {
            panic!("the mac is gone")
        };
        assert_eq!(*xrf_read_incr, 7);
        assert_eq!(body.len(), 2);
        let Op::Sentient(sen::Op::ScalarAdd { lhs, rhs, .. }) = &body[1] else {
            panic!("the second add is gone")
        };
        assert_eq!((*lhs, *rhs), (Val(1), Val(1)));
    }

    /// THREE OPS' WORTH OF UNROLL SIZE ROUNDS DOWN TO AN `x2` AND ONE `x1` CLONE — the clones ARE the
    /// port, so a statement standing for three ops leaves two behind.
    #[test]
    fn setting_unroll_fields_clones_the_statement_until_the_unroll_size_is_spent() {
        let mut body = vec![Op::Sentient(sen::Op::VectorUnary {
            mask: Val(0),
            op_a: Operand {
                port_id: Some(0),
                ..Operand::from(sen::Port::West)
            },
            unary_op: UnaryOp::Rec,
            result: ResultPorts::default(),
            compute_precision: Precision::Fp16,
            fold_mode: None,
            unroll_factor: UnrollFactor::X1,
            dbg_name: None,
        })];
        let mut operand_list = UnrollOperands::default();
        operand_list.op_name = Some(RolledOp::Unary(UnaryOp::Rec));
        operand_list.are_unroll_fields_updated = true;
        operand_list.unroll_size = SnapshotSize(3);
        operand_list.operand_list[OperandName::OpA.slot()] = Some(sen::Port::West);
        let mut values = Values::default();
        let rerolled = set_unroll_fields_in_stmt(
            &mut body,
            0,
            &operand_list,
            DfirUnit::Sfp,
            SenTarget::Sentient,
            &mut values,
        );
        assert_eq!(rerolled, Rerolled::Done);
        assert_eq!(body.len(), 2);
        let factors: Vec<UnrollFactor> = body
            .iter()
            .map(|op| {
                let Op::Sentient(sen::Op::VectorUnary { unroll_factor, .. }) = op else {
                    panic!("not a unary")
                };
                *unroll_factor
            })
            .collect();
        assert_eq!(factors, vec![UnrollFactor::X2, UnrollFactor::X1]);
    }
}
