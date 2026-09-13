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

// ⛔ NOTHING IN THE CRATE CALLS THIS FILE UNTIL `e607_runOnOperation` (level 5) LANDS, and CI runs
// clippy with `-D warnings`. ⭐ REMOVE THIS WITH e607.
#![allow(dead_code)]

pub(crate) mod unroll_operands;

use crate::arch::{Arch, IsaGen};
use crate::islands::dataflow_ir::dialects::dataflow;
use crate::islands::dataflow_ir::ty::GenericComp;
use crate::islands::sentient::ProgramUnit;
use crate::islands::sentient::dialects::{
    Op, Val, operands, regions_mut, regions_ref, replace_all_uses_with, results, sentient,
    use_count,
};
use crate::transform::sentient::op_rerolling::unroll_operands::{UnrollOperands, XrfIncr};
use crate::transform::sentient::utils::{InBlock, OpAt, fold_mode_attribute_if_exists};
use crate::units::DfirUnit;

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
    /// A single instance — the declaration's initial value, and what
    /// [`unroll_operands::UnrollOperands::reset`] restores; for a memory op the count is the burst
    /// size instead (`OpRerolling.cpp:643`).
    pub const ONE: Self = Self(1);

    /// Replaces: e122_incrementUnrollSize
    ///
    /// Adds `incr_val` further ops to the count this rerolled statement stands for
    /// (`dcc/src/Transform/Sentient/OpRerolling.hpp:131`).
    pub fn increment(&mut self, incr_val: UnrollSize) {
        // `unroll_size_ += incr_val;` — `unsigned`, so the reference wraps rather than trapping.
        self.0 = self.0.wrapping_add(incr_val.0);
    }
}

// crustify:todo: e334_mergeScalarOpIntoMac
//   authority : dcc/src/Transform/Sentient/OpRerolling.cpp:81  (57 body lines, level 1)
//   original  : void OpRerollingPass::mergeScalarOpIntoMac(Block *bb)
//   calls     : e252_size

// crustify:todo: e338_setUnrollFieldsInStmt
//   authority : dcc/src/Transform/Sentient/OpRerolling.cpp:1038  (311 body lines, level 1)
//   original  : void OpRerollingPass::setUnrollFieldsInStmt(Operation *op, UnrollOperands &operand_list, SenComponents type)
//   calls     : e113_isXrfRdOp, e114_isXrfWtOp, e116_getUnrollOperandUsingPort, e119_createOperand, e120_createForwardingArray, e243_roundDownUnrollFactor, e252_size

/// Replaces: e453_updateRefOpUnrollInfo
///
/// Merges `curr_op` into the rerolled statement `ref_op` stands for — the unroll info, the size, the
/// XRF write increment, `curr_op`'s uses and its deletion — and closes that statement when `curr_op`
/// was the block's last op or carried an XRF read increment.
///
/// ⛔ TRAP: `ref_op = nullptr` AND `prev_op = curr_op` ARE DEAD STORES. Both are pointers taken BY
/// VALUE, so neither reaches the caller; `prev_op` is never read at all and is therefore not a
/// parameter here, and e519 has to keep its own two.
/// ⛔ TRAP: A SPLAT'S PORTS GO ON THE DELETE LIST WHILE THE SPLAT STILL USES THEM, so `hasOneUse`
/// means *used by this splat only* — asking after the erase would answer zero.
/// ⭐ THE TWO OPS' RESULTS PAIR POSITIONALLY and the counts are equal because `e336_match` admitted
/// the pair; a shorter `ref_op` indexes past its own results in the reference.
pub(crate) fn update_ref_op_unroll_info(
    block: &mut [Op],
    curr_op: InBlock,
    ref_op: InBlock,
    last_op: InBlock,
    ref_operand_list: &mut UnrollOperands,
    curr_operand_list: &UnrollOperands,
    ops_to_be_deleted: &mut Vec<InBlock>,
    ty: DfirUnit,
) {
    // `is_field_unroll_` is updated once, at the beginning.
    ref_operand_list.update_unroll_info(curr_operand_list);
    ref_operand_list
        .unroll_size
        .increment(curr_operand_list.unroll_size);
    if ref_operand_list.is_xrf_wt_op() {
        // `int + int` — wrapping for the reason `UnrollSize::increment` gives.
        ref_operand_list.xrf_write_incr = XrfIncr(
            ref_operand_list
                .xrf_write_incr
                .0
                .wrapping_add(curr_operand_list.xrf_write_incr.0),
        );
    }
    // `curr_op`'s uses become `ref_op`'s.
    let pairs: Vec<(Val, Val)> = results(&block[curr_op.0])
        .into_iter()
        .zip(results(&block[ref_op.0]))
        .collect();
    for (of, with) in pairs {
        replace_all_uses_with(block, of, with);
    }
    ops_to_be_deleted.push(curr_op);
    let splat_ports = match &block[curr_op.0] {
        Op::Sentient(sentient::Op::Splat { input, output, .. }) => Some((*input, *output)),
        _ => None,
    };
    if let Some((input, output)) = splat_ports {
        for port in [input, output] {
            if let Some(at) = logical_port_used_once(block, port) {
                ops_to_be_deleted.push(at);
            }
        }
    }
    // Is `curr_op` the last op in the region?
    if curr_op == last_op
        || (curr_operand_list.is_xrf_rd_op() && curr_operand_list.xrf_read_incr != XrfIncr::ZERO)
    {
        ref_operand_list.xrf_read_incr = curr_operand_list.xrf_read_incr;
        set_unroll_fields_in_stmt(&mut block[ref_op.0], ref_operand_list, ty);
        ref_operand_list.reset();
    }
}

/// Replaces: e454_updateRefOpUnrollFields
///
/// Closes the rerolled statement `ref_op` stands for, then starts a new one from `curr_op`'s snapshot
/// unless that snapshot is the `"NA"` sentinel.
///
/// ⛔⛔ NOTHING IN `dcc/` CALLS THIS: `processOneBlock` (e519) inlines the identical statements three
/// times over instead, so the unit is live code that is never entered.
/// ⛔ TRAP: `ref_op`, `prev_op` AND THE `curr_op` THEY TAKE THEIR VALUE FROM ARE ALL DEAD STORES —
/// pointers by value again — so the only effects that leave here are on `ref_operand_list`, and
/// `curr_op` is not a parameter.
/// ⭐ `getOpName().contains("NA")` IS THE ABSENT SNAPSHOT: no [`unroll_operands::RolledOp`] spelling
/// holds an upper-case `NA`, so the substring test is exactly `op_name: None`.
pub(crate) fn update_ref_op_unroll_fields(
    block: &mut [Op],
    ref_op: InBlock,
    ref_operand_list: &mut UnrollOperands,
    curr_operand_list: &UnrollOperands,
    ty: DfirUnit,
) {
    set_unroll_fields_in_stmt(&mut block[ref_op.0], ref_operand_list, ty);
    ref_operand_list.reset();
    // Only copy when the op is a legal type.
    if curr_operand_list.op_name.is_some() {
        ref_operand_list.assign_from(curr_operand_list);
    }
}

/// `input->hasOneUse()` FOR THE `sentient.logical_port` DEFINING `val` — where it sits in the block,
/// and `None` when the definer is anything else or the value has another reader.
///
/// ⭐ THE OP'S USE COUNT IS THE VALUE'S: a `sentient.logical_port` binds exactly one result.
/// ⛔ THE REFERENCE ASSERTS WHERE THIS ANSWERS `None`: `isa<>(nullptr)` on a splat port with no
/// defining op is an assertion failure, not a `false`.
fn logical_port_used_once(block: &[Op], val: Val) -> Option<InBlock> {
    let at = block.iter().position(
        |op| matches!(op, Op::Sentient(sentient::Op::LogicalPort { result, .. }) if *result == val),
    )?;
    (use_count(val, block) == 1).then_some(InBlock(at))
}

/// `OpRerollingPass::setUnrollFieldsInStmt(Operation *, UnrollOperands &, SenComponents)` — the callee
/// e453 and e454 share, and SENPASS UNIT e338, whose anchor is still open above.
///
/// ⛔ e338 IS A LEVEL-1 DEPENDENCY THAT NO REMAINING SCHEDULE OWNS: sc2's port driver died on an
/// authentication error with 12 batches unrun, `sentient.cpp: e334_mergeScalarOpIntoMac +7` among
/// them. Isolating the call in a seam is the `2a8195231` precedent, and e338's TODO is left untouched
/// — filling it is not this batch's work.
fn set_unroll_fields_in_stmt(op: &mut Op, operand_list: &mut UnrollOperands, ty: DfirUnit) {
    let _ = op;
    todo!(
        "OpRerollingPass::setUnrollFieldsInStmt — senpass e338 (OpRerolling.cpp:1038) is not ported \
         yet, and this {ty:?} statement's {:?} of unroll size {:?} needs it",
        operand_list.op_name,
        operand_list.unroll_size
    )
}

/// `dccExtContext().getArch() == MPW4_ISA` (`:249-257`) — the gate on the DD1 hardware-bug workaround
/// that never rerolls an XRF-writing mac, and ⛔ A CONSTANT FALSE HERE: [`crate::arch::IsaGen`] names
/// `RCUDD1A` and `SEN1P5` only, *"a generation no build can select cannot arrive here"*
/// (`src/arch.rs:20-22`), so no program this crate compiles takes that path. Its other half,
/// `MacOp::isXrfWtRelated()`, is ported at
/// [`crate::bridges::dataflow_ir_to_sentient::vc_lowering_xrf::MacXrfIncrements::wt_related`].
const FORCE_RESET_ON_MPW4_XRF_WRITE: bool = false;

/// Replaces: e519_processOneBlock
///
/// Rerolls one block: every op is snapshotted, matched against the statement being built, and either
/// merged into it or made the start of the next one — with a `fold_AB_A`/`fold_AB_B` pair carried as
/// TWO statements at once.
///
/// ⛔ TRAP: THE FOLD PAIR IS FOUND THROUGH `getNextNode()` (`:266-273`), not through the next
/// candidate, so a `sentient.scalar_constant` between the halves hides the pair the loop then skips.
/// ⛔ TRAP: THE STATEMENT IS CLOSED WITH A `ref_op` THAT MAY BE NULL (`:302-311`) — see
/// [`set_unroll_fields_in_ref_op`], which is why e519 stops at e335 and not at e338.
/// ⭐ `prev_op_pair` IS WRITTEN SIX TIMES AND NEVER READ — the dead store e453's own note names.
pub(crate) fn process_one_block(ty: DfirUnit, block: &mut Vec<Op>) {
    // `auto &last_op = bb->back();` — an empty block has no back to take.
    let Some(last_op) = block.len().checked_sub(1).map(InBlock) else {
        return;
    };
    let mut curr_op_pair: (Option<InBlock>, Option<InBlock>) = (None, None);
    let mut ref_op_pair: (Option<InBlock>, Option<InBlock>) = (None, None);
    let mut prev_op_pair: (Option<InBlock>, Option<InBlock>) = (None, None);
    let mut ref_list = (UnrollOperands::default(), UnrollOperands::default());
    let mut ops_to_be_deleted: Vec<InBlock> = Vec::new();
    let mut fold_ab_mode = false;
    let mut skip_next = false;
    for at in 0..block.len() {
        if is_stepped_over(&block[at]) {
            continue;
        }
        // The DD1 workaround's `force_reset_ref_op`, computed here as the reference does.
        let force_reset_ref_op = FORCE_RESET_ON_MPW4_XRF_WRITE;
        if skip_next {
            skip_next = false;
            continue;
        }
        // `fold_AB_mode` is the value from the last iteration.
        let was_last_fold_mode_ab = fold_ab_mode;
        fold_ab_mode = false;
        let curr = InBlock(at);
        curr_op_pair.0 = Some(curr);
        let curr_op_a_fold = fold_mode_attribute_if_exists(&block[at]);
        let curr_op_b_fold = block.get(at + 1).and_then(fold_mode_attribute_if_exists);
        if curr_op_a_fold == Some(sentient::FoldMode::FoldAbA)
            && curr_op_b_fold == Some(sentient::FoldMode::FoldAbB)
        {
            fold_ab_mode = true;
            skip_next = true;
            curr_op_pair.1 = Some(InBlock(at + 1));
        }
        // A fold pair opening or closing ends whatever statement was being built (`:300-315`).
        if was_last_fold_mode_ab != fold_ab_mode {
            set_unroll_fields_in_ref_op(block, ref_op_pair.0, &mut ref_list.0, ty);
            if was_last_fold_mode_ab {
                set_unroll_fields_in_ref_op(block, ref_op_pair.1, &mut ref_list.1, ty);
                ref_list.0.reset();
                ref_list.1.reset();
            } else {
                ref_list.0.reset();
            }
        }

        // No statement to match against yet: this op — or this pair — starts one (`:317-347`).
        if fold_ab_mode {
            if ref_op_pair.0.is_none() || ref_op_pair.1.is_none() || !was_last_fold_mode_ab {
                ref_list.0.fill(&block[at], ty);
                if let Some(second) = curr_op_pair.1 {
                    ref_list.1.fill(&block[second.0], ty);
                }
                if ref_list.0.op_name.is_none() || ref_list.1.op_name.is_none() {
                    ref_list.0.reset();
                    ref_list.1.reset();
                } else {
                    ref_op_pair = curr_op_pair;
                    prev_op_pair = curr_op_pair;
                }
                continue;
            }
        } else if ref_op_pair.0.is_none() || was_last_fold_mode_ab {
            ref_list.0.fill(&block[at], ty);
            if ref_list.0.op_name.is_none() {
                ref_list.0.reset();
            } else {
                ref_op_pair.0 = curr_op_pair.0;
                prev_op_pair.0 = curr_op_pair.0;
            }
            continue;
        }

        // If the op name and precisions are the same and the operands match, this op joins the
        // statement; otherwise the statement is closed and this op starts the next one.
        let mut curr_list = (UnrollOperands::default(), UnrollOperands::default());
        curr_list.0.fill(&block[at], ty);
        if fold_ab_mode {
            if let Some(second) = curr_op_pair.1 {
                curr_list.1.fill(&block[second.0], ty);
            }
        }

        if was_last_fold_mode_ab {
            let pair_rerolls = fold_ab_mode
                && is_user(block, ref_op_pair.0, curr_op_pair.0, &ops_to_be_deleted)
                && is_user(block, ref_op_pair.1, curr_op_pair.1, &ops_to_be_deleted)
                && ref_list.0.matches(&curr_list.0)
                && ref_list.1.matches(&curr_list.1)
                && !force_reset_ref_op;
            if pair_rerolls {
                merge_into_ref_op(
                    block,
                    ty,
                    curr_op_pair.0,
                    ref_op_pair.0,
                    last_op,
                    &mut ref_list.0,
                    &curr_list.0,
                    &mut ops_to_be_deleted,
                );
                merge_into_ref_op(
                    block,
                    ty,
                    curr_op_pair.1,
                    ref_op_pair.1,
                    last_op,
                    &mut ref_list.1,
                    &curr_list.1,
                    &mut ops_to_be_deleted,
                );
            } else {
                close_and_restart(
                    block,
                    ty,
                    &mut ref_op_pair.0,
                    &mut prev_op_pair.0,
                    &mut ref_list.0,
                    curr_op_pair.0,
                    &curr_list.0,
                );
                close_and_restart(
                    block,
                    ty,
                    &mut ref_op_pair.1,
                    &mut prev_op_pair.1,
                    &mut ref_list.1,
                    curr_op_pair.1,
                    &curr_list.1,
                );
            }
        } else if !fold_ab_mode
            && is_user(block, ref_op_pair.0, curr_op_pair.0, &ops_to_be_deleted)
            && ref_list.0.matches(&curr_list.0)
            && !force_reset_ref_op
        {
            merge_into_ref_op(
                block,
                ty,
                curr_op_pair.0,
                ref_op_pair.0,
                last_op,
                &mut ref_list.0,
                &curr_list.0,
                &mut ops_to_be_deleted,
            );
        } else {
            close_and_restart(
                block,
                ty,
                &mut ref_op_pair.0,
                &mut prev_op_pair.0,
                &mut ref_list.0,
                curr_op_pair.0,
                &curr_list.0,
            );
        }
    }
    // `dropAllUses(); erase();` (`:427-434`) — an op leaves by POSITION, so the queue leaves
    // highest-first, and dropping the uses is what removing the definition is here.
    // ⭐ THE REFERENCE ONLY SKIPS A **CONSECUTIVE** REPEAT; a non-consecutive one would be erased
    // twice, and cannot arise because e453 queues a logical port only while nothing else reads it.
    let mut doomed: Vec<usize> = ops_to_be_deleted.iter().map(|op| op.0).collect();
    doomed.sort_unstable();
    doomed.dedup();
    for at in doomed.into_iter().rev() {
        block.remove(at);
    }
}

/// `isa<sentient::ConstantOp, dataflow::GetUnitOp, sentient::LogicalPortOp>` (`:232-234`) — the three
/// ops the walk steps over without closing anything.
///
/// ⭐ `sentient::ConstantOp` IS `sentient.scalar_constant` (`SentientOps.td:848`) and NOT the vector
/// one, which is a candidate like any other op.
fn is_stepped_over(op: &Op) -> bool {
    matches!(
        op,
        Op::Sentient(sentient::Op::ScalarConstant { .. } | sentient::Op::LogicalPort { .. })
            | Op::Dataflow(dataflow::Op::GetUnit { .. })
    )
}

/// `setUnrollFieldsInStmt(ref_op, ref_operand_list, type)` WHERE `ref_op` MAY BE NULL.
///
/// ⭐⭐ e338'S OWN FIRST TWO STATEMENTS RETURN FOR AN ABSENT SNAPSHOT OR UN-UPDATED FIELDS
/// (`:1041-1043`) — BEFORE it reads `op` — which is how the reference survives being handed a null
/// `ref_op` at `:302`, `:308` and `:369`. Asking that question here is what keeps the seam a rerolled
/// program actually reaches visible: e519 stops at e335, not at e338.
/// ⛔ DIVERGENCE: past that guard the reference would dereference the null; there is nothing to write
/// to instead, so nothing is written.
fn set_unroll_fields_in_ref_op(
    block: &mut [Op],
    ref_op: Option<InBlock>,
    ref_operand_list: &mut UnrollOperands,
    ty: DfirUnit,
) {
    if ref_operand_list.op_name.is_none() || !ref_operand_list.are_unroll_fields_updated {
        return;
    }
    let Some(ref_op) = ref_op else {
        return;
    };
    set_unroll_fields_in_stmt(&mut block[ref_op.0], ref_operand_list, ty);
}

/// [`update_ref_op_unroll_info`] FOR A PAIR HALF THAT MAY BE ABSENT — the reference passes raw
/// pointers, and both are non-null on every path that reaches the call.
fn merge_into_ref_op(
    block: &mut [Op],
    ty: DfirUnit,
    curr_op: Option<InBlock>,
    ref_op: Option<InBlock>,
    last_op: InBlock,
    ref_operand_list: &mut UnrollOperands,
    curr_operand_list: &UnrollOperands,
    ops_to_be_deleted: &mut Vec<InBlock>,
) {
    let (Some(curr_op), Some(ref_op)) = (curr_op, ref_op) else {
        return;
    };
    update_ref_op_unroll_info(
        block,
        curr_op,
        ref_op,
        last_op,
        ref_operand_list,
        curr_operand_list,
        ops_to_be_deleted,
        ty,
    );
}

/// THE `else` ARM `processOneBlock` WRITES OUT THREE TIMES (`:365-397`, `:406-419`) — close the
/// statement, then start the next one from `curr_op`'s snapshot, or from nothing where that snapshot
/// is the `"NA"` sentinel.
///
/// ⭐ THIS IS e454 WITH ITS DEAD STORES MADE LIVE, which is exactly what that unit's note says the
/// reference does instead of calling it: `ref_op` and `prev_op` reach the caller here.
fn close_and_restart(
    block: &mut [Op],
    ty: DfirUnit,
    ref_op: &mut Option<InBlock>,
    prev_op: &mut Option<InBlock>,
    ref_operand_list: &mut UnrollOperands,
    curr_op: Option<InBlock>,
    curr_operand_list: &UnrollOperands,
) {
    set_unroll_fields_in_ref_op(block, *ref_op, ref_operand_list, ty);
    ref_operand_list.reset();
    // Only copy when the op is a legal type.
    if curr_operand_list.op_name.is_none() {
        *ref_op = None;
        *prev_op = None;
    } else {
        *ref_op = curr_op;
        *prev_op = curr_op;
        ref_operand_list.assign_from(curr_operand_list);
    }
}

/// `is_user(prev_op, curr_op)` (`:212-230`) — whether `curr_op` is the only reader of `prev_op` left,
/// counting the ops already queued for deletion as gone.
///
/// ⛔ TRAP: ITS `prev_op` PARAMETER IS PASSED `ref_op` AT BOTH CALL SITES (`:355`, `:399`), so the
/// question is about the statement being built and not about the previous op at all.
/// ⛔ TRAP: THE EMPTY-QUEUE TEST IS REDUNDANT (`:224-228`) — `find` over an empty list already fails.
/// ⭐ A READER INSIDE A REGION IS NEITHER `curr_op` NOR DELETABLE, so it refuses the merge.
fn is_user(
    block: &[Op],
    prev_op: Option<InBlock>,
    curr_op: Option<InBlock>,
    ops_to_be_deleted: &[InBlock],
) -> bool {
    let (Some(prev_op), Some(curr_op)) = (prev_op, curr_op) else {
        return false;
    };
    let produced = results(&block[prev_op.0]);
    // Both without results is a merge; a `prev_op` without results and a `curr_op` with them is not.
    if produced.is_empty() {
        return results(&block[curr_op.0]).is_empty();
    }
    let mut any_user = false;
    for (at, op) in block.iter().enumerate() {
        if regions_ref(op)
            .iter()
            .any(|region| block_reads(region, &produced))
        {
            return false;
        }
        if !operands(op).iter().any(|val| produced.contains(val)) {
            continue;
        }
        any_user = true;
        let user = InBlock(at);
        if user != curr_op && !ops_to_be_deleted.contains(&user) {
            return false;
        }
    }
    // `prev_op->use_empty()` — results nobody reads are not a statement to extend.
    any_user
}

/// Whether anything in a region — at any depth — reads one of `produced`.
fn block_reads(block: &[Op], produced: &[Val]) -> bool {
    block.iter().any(|op| {
        operands(op).iter().any(|val| produced.contains(val))
            || regions_ref(op)
                .iter()
                .any(|region| block_reads(region, produced))
    })
}

/// `mergeScalarOpIntoMac(unit_op.getBody())` — e334, not ported yet.
fn merge_scalar_op_into_mac(unit_body: &mut Vec<Op>) {
    todo!(
        "OpRerollingPass::mergeScalarOpIntoMac — senpass e334 (OpRerolling.cpp:81) is not ported \
         yet, and this PT unit's {} rerolled ops need it",
        unit_body.len()
    )
}

/// `bool merge_xrf_into_mac` — ⛔ A BOOL PARAMETER IS NOT A CRATE TYPE, and its two call sites state
/// opposite answers (`:1004` true, `MultiDimLoopPeeling.cpp:676` false).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MergeXrfIntoMac {
    /// The pass's own entry, which merges.
    Yes,
    /// Loop peeling's call on one peeled loop, which does not.
    No,
}

/// WHAT `op->walk` COVERS — the whole unit or one op inside it.
///
/// ⛔ THE WALKED SCOPE AND THE OWNING UNIT ARE SEPARATE, because a path cannot be reborrowed out of
/// the unit it indexes: `op->getParentOfType<ProgramUnitOp>()` (`:1009-1012`) is the caller's fact.
#[derive(Debug, Clone)]
pub(crate) enum RerollScope {
    /// `isa<dataflow::ProgramUnitOp>(op)` — the unit body itself (`:1004`).
    WholeUnit,
    /// An op inside it, whose own regions are what gets walked (`MultiDimLoopPeeling.cpp:676`).
    Op(OpAt),
}

/// Replaces: e571_runOpRerolling
///
/// Rerolls every block of the scope, innermost first, on the nine components that reroll at all, then
/// merges a PT unit's scalar ops into the MAC ahead of them (`:1008-1036`).
///
/// ⛔ TRAP: THE DD2 LDST BUG IS A `return`, NOT A SKIP (`:1024-1028`) — on `RCUDD1A` an L0, L0LU or
/// L0SU unit leaves this having done nothing at all, the PT tail included.
/// ⭐ `getUnitType` IS THE GENERIC COMPONENT, so a PT row and a PT row span are both `PT` here.
/// ⛔ `DT_CHECK(unit_op.getUnits().size() >= 1)` IS DISCHARGED BY [`crate::islands::dataflow_ir::Units`].
pub(crate) fn run_op_rerolling<A: Arch>(
    unit: &mut ProgramUnit<A>,
    scope: &RerollScope,
    merge_xrf_into_mac: MergeXrfIntoMac,
) {
    let unit_type = unit.on.kind();
    let generic = unit_type.generic();
    let handled = matches!(
        generic,
        GenericComp::Sfp
            | GenericComp::Pe
            | GenericComp::Pt
            | GenericComp::Lx
            | GenericComp::L0
            | GenericComp::Lxlu
            | GenericComp::Lxsu
            | GenericComp::L0lu
            | GenericComp::L0su
    );
    let dd2_ldst_bug = matches!(A::GEN, IsaGen::Rcudd1a)
        && matches!(
            generic,
            GenericComp::L0 | GenericComp::L0lu | GenericComp::L0su
        );
    if handled && !dd2_ldst_bug {
        match scope {
            RerollScope::WholeUnit => reroll_blocks(unit_type, &mut unit.body),
            RerollScope::Op(at) => {
                if let Some(op) = at.op_mut(&mut unit.body) {
                    for region in regions_mut(op) {
                        reroll_blocks(unit_type, region);
                    }
                }
            }
        }
    }
    // after rerolling ops, merge scalar_add/sub to its preceding mac
    if matches!(generic, GenericComp::Pt) && matches!(merge_xrf_into_mac, MergeXrfIntoMac::Yes) {
        merge_scalar_op_into_mac(&mut unit.body);
    }
}

/// The `walk<WalkOrder::PostOrder>` over blocks (`:1029-1030`): the nested ones first.
fn reroll_blocks(ty: DfirUnit, block: &mut Vec<Op>) {
    for at in 0..block.len() {
        for region in regions_mut(&mut block[at]) {
            reroll_blocks(ty, region);
        }
    }
    process_one_block(ty, block);
}

// crustify:todo: e607_runOnOperation
//   authority : dcc/src/Transform/Sentient/OpRerolling.cpp:994  (13 body lines, level 5)
//   original  : void OpRerollingPass::runOnOperation()
//   calls     : e571_runOpRerolling

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Dd2;
    use crate::islands::dataflow_ir::Units;
    use crate::islands::sentient::dialects::sentient::{LrfIndex, Port, Precision, SplatPad};
    use crate::units::Row;

    /// AN UNREROLLED STATEMENT ALREADY STANDS FOR ONE OP, and each merge adds its own count.
    #[test]
    fn incrementing_adds_to_a_size_that_starts_at_one() {
        let mut size = UnrollSize::default();
        assert_eq!(size, UnrollSize(1));
        size.increment(UnrollSize(3));
        assert_eq!(size, UnrollSize(4));
    }

    /// `%r = sentient.logical_port {portName = port}`.
    fn logical_port(result: u32, port: Port) -> Op {
        Op::Sentient(sentient::Op::LogicalPort {
            port_name: port,
            result: Val(result),
        })
    }

    /// `sentient.splat %input, %output`.
    fn splat(input: u32, output: u32) -> Op {
        Op::Sentient(sentient::Op::Splat {
            input: Val(input),
            output: Val(output),
            mask: Val(99),
            pad: SplatPad::None,
            precision: Precision::Fp16,
            program_header: false,
            unroll_factor: sentient::UnrollFactor::X1,
            unroll_incr_result: false,
            dbg_name: None,
        })
    }

    /// e453's splat rule — the two logical ports a splat owns are the ones queued for deletion with
    /// it, and only while nothing else reads them.
    #[test]
    fn e453_queues_only_the_logical_ports_the_splat_alone_reads() {
        let owned = vec![
            logical_port(1, Port::Lrf(LrfIndex::L0)),
            logical_port(2, Port::Lrf(LrfIndex::L1)),
            splat(1, 2),
        ];
        assert_eq!(logical_port_used_once(&owned, Val(1)), Some(InBlock(0)));
        assert_eq!(logical_port_used_once(&owned, Val(2)), Some(InBlock(1)));

        // ⭐ A SECOND READER KEEPS THE INPUT PORT ALIVE, which is the whole of `hasOneUse()`.
        let mut shared = owned.clone();
        shared.push(splat(1, 3));
        assert_eq!(logical_port_used_once(&shared, Val(1)), None);
        assert_eq!(logical_port_used_once(&shared, Val(2)), Some(InBlock(1)));

        // A port this block does not define is the reference's null `getDefiningOp()`.
        assert_eq!(logical_port_used_once(&owned, Val(7)), None);
    }

    /// e453 — the merge's FIRST statement is the unroll-info update, which is e337, and e337 is not
    /// ported, so nothing after it is reachable yet.
    #[test]
    #[should_panic(expected = "senpass e337")]
    fn e453_starts_at_the_unported_e337() {
        let mut block = vec![splat(1, 2)];
        let mut reference = UnrollOperands::default();
        let candidate = UnrollOperands::default();
        let mut to_delete = Vec::new();
        update_ref_op_unroll_info(
            &mut block,
            InBlock(0),
            InBlock(0),
            InBlock(0),
            &mut reference,
            &candidate,
            &mut to_delete,
            DfirUnit::Pe,
        );
    }

    /// `%c = sentient.scalar_constant {value = 0 : si64} : index`.
    fn scalar_constant(result: u32) -> Op {
        Op::Sentient(sentient::Op::ScalarConstant {
            value: 0,
            result: Val(result),
            reg_locale: sentient::RegType::Imm,
            ty: crate::islands::dataflow_ir::ty::ScalarTy::Index,
            is_symbol: false,
        })
    }

    /// `%u = dataflow.get_unit {type = pe} : index`.
    fn get_unit(result: u32) -> Op {
        Op::Dataflow(dataflow::Op::GetUnit {
            result: Val(result),
            residency: crate::units::Residency::Global,
            unit: DfirUnit::Pe,
            num_folds: None,
        })
    }

    /// e519 — THE WALK STEPS OVER THE THREE OPS IT IS TOLD TO and snapshots the first real candidate,
    /// which is e335. ⭐ REACHING THAT SEAM IS WHAT IS TESTABLE: every path through the block bottoms
    /// out in it at the first candidate, so a program cannot be rerolled until e335 lands.
    #[test]
    #[should_panic(expected = "senpass e335")]
    fn e519_snapshots_the_first_candidate_at_the_unported_e335() {
        let mut block = vec![
            get_unit(0),
            logical_port(1, Port::Lrf(LrfIndex::L0)),
            scalar_constant(2),
            splat(1, 3),
        ];
        process_one_block(DfirUnit::Pe, &mut block);
    }

    /// e519 — a block of nothing but stepped-over ops closes no statement and deletes nothing, which
    /// is the negative the skip list exists for: `ref_op` stays null and e338's own guard is what
    /// keeps that harmless.
    #[test]
    fn e519_leaves_a_block_of_stepped_over_ops_alone() {
        let untouched = vec![
            get_unit(0),
            logical_port(1, Port::Lrf(LrfIndex::L0)),
            scalar_constant(2),
        ];
        let mut block = untouched.clone();
        process_one_block(DfirUnit::Pe, &mut block);
        assert_eq!(block, untouched);
    }

    /// e454 — closing the statement is e338, which is not ported, so the reset and the copy after it
    /// are not reachable yet either.
    #[test]
    #[should_panic(expected = "senpass e338")]
    fn e454_starts_at_the_unported_e338() {
        let mut block = vec![splat(1, 2)];
        let mut reference = UnrollOperands::default();
        let candidate = UnrollOperands::default();
        update_ref_op_unroll_fields(
            &mut block,
            InBlock(0),
            &mut reference,
            &candidate,
            DfirUnit::Pe,
        );
    }
    /// 571/656 — a PT unit is one of the nine that reroll, so the walk runs over its blocks (this one
    /// closes no statement) and the PT tail is then reached, which is the unported e334.
    #[test]
    #[should_panic(expected = "senpass e334")]
    fn e571_walks_then_reaches_the_pt_merge_tail() {
        let mut unit: ProgramUnit<Dd2> = ProgramUnit {
            on: Units::one(DfirUnit::PtRow(Row::checked(0).expect("row 0")), Val(0)),
            precision: None,
            body: vec![get_unit(0), logical_port(1, Port::Lrf(LrfIndex::L0))],
            arch: core::marker::PhantomData,
        };
        run_op_rerolling(&mut unit, &RerollScope::WholeUnit, MergeXrfIntoMac::Yes);
    }
}
