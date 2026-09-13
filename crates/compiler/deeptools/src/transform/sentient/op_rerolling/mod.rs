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

pub(crate) mod unroll_operands;

use core::num::NonZeroU32;

use crate::arch::{Arch, Elements, IsaGen};
use crate::bridges::dataflow_ir_to_sentient::vc_lowering_xrf::MacXrfIncrements;
use crate::islands::dataflow_ir::dialects::dataflow;
use crate::islands::dataflow_ir::ty::GenericComp;
use crate::islands::dataflow_ir::{ValueMapping, Values};
use crate::islands::sentient::{Program, ProgramUnit};
use crate::islands::sentient::dialects::{
    Op, Val, clone_ops, defining_op, operands, regions_mut, regions_ref, replace_all_uses_with,
    results, sentient, use_count,
};
use crate::model::Model;
use crate::transform::sentient::analyses::InstructionEstimator;
use crate::transform::sentient::op_rerolling::unroll_operands::{
    ComputePortId, OperandName, UnrollOperands, XrfIncr,
};
use crate::transform::sentient::utils::{
    InBlock, OpAt, SenTarget, fold_mode_attribute_if_exists, round_down_unroll_factor,
};
use crate::units::DfirUnit;
use crate::workload::Workload;

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
#[allow(clippy::too_many_arguments)]
pub(crate) fn update_ref_op_unroll_info(
    block: &mut [Op],
    curr_op: InBlock,
    ref_op: InBlock,
    last_op: InBlock,
    ref_operand_list: &mut UnrollOperands,
    curr_operand_list: &UnrollOperands,
    ops_to_be_deleted: &mut Vec<InBlock>,
    ty: DfirUnit,
    closing: &mut Closing<'_>,
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
        set_unroll_fields_in_stmt(block, ref_op, ref_operand_list, ty, closing);
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
    closing: &mut Closing<'_>,
) {
    set_unroll_fields_in_stmt(block, ref_op, ref_operand_list, ty, closing);
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

/// EVERYTHING [`set_unroll_fields_in_stmt`] NEEDS THAT IS NOT THE OP OR ITS SNAPSHOT — the pass
/// state the reference reaches through `dccExtContext()` and its `OpBuilder`.
#[derive(Debug)]
pub(crate) struct Closing<'a> {
    /// `dccExtContext().dsc_global_->backend`, which [`round_down_unroll_factor`] reads.
    pub(crate) sen_target: SenTarget,
    /// Where a clone's fresh results come from.
    pub(crate) values: &'a mut Values,
    /// Where the clones wait; see [`PendingClones`].
    pub(crate) pending: PendingClones,
}

/// THE OPS `builder.setInsertionPointAfter(op)` WOULD HAVE PUT IN THE BLOCK ALREADY, HELD BACK.
///
/// ⛔⛔ INSERTING DURING THE WALK WOULD MOVE EVERY LATER OP, and e519's whole state is POSITIONS:
/// `curr_op`, `ref_op`, `last_op`, `prev_op` and the delete queue are all [`InBlock`] indices, where
/// the reference held intrusive-list pointers that no insertion disturbs. So the clones are collected
/// against the position they follow and spliced in once, together with the deletions.
///
/// ⭐ NOTHING OBSERVES THE DIFFERENCE. `ref_op` is always behind the walk cursor, so the reference's
/// own iterator never revisits what it inserted either; a clone can only read values defined at or
/// before the op it copies; and an op e338 ran on is never itself queued for deletion.
#[derive(Debug, Default)]
pub(crate) struct PendingClones(Vec<(usize, Vec<Op>)>);

impl PendingClones {
    /// The ops to place directly after position `at`, in the order they were built.
    fn push_after(&mut self, at: InBlock, ops: Vec<Op>) {
        if ops.is_empty() {
            return;
        }
        if let Some(entry) = self.0.iter_mut().find(|(position, _)| *position == at.0) {
            entry.1.extend(ops);
        } else {
            self.0.push((at.0, ops));
        }
    }

    /// Hands over what follows position `at`, leaving nothing behind.
    fn take(&mut self, at: InBlock) -> Vec<Op> {
        self.0
            .iter_mut()
            .find(|(position, _)| *position == at.0)
            .map_or_else(Vec::new, |entry| core::mem::take(&mut entry.1))
    }
}

/// `builder.clone(*op)` FOR ONE OP — [`clone_ops`] over a one-element slice, which mints fresh results
/// and leaves every operand pointing where the original's did.
fn cloned(op: &Op, values: &mut Values) -> Option<Op> {
    clone_ops(core::slice::from_ref(op), values, &mut ValueMapping::new()).pop()
}

/// e338's `round_down` lambda (`:1048-1059`).
///
/// ⛔ `type == L3` IS UNSTATABLE AND UNREACHABLE: [`DfirUnit`] has no plain `L3` — only `L3lu`,
/// `L3su`, `L3Ibr` — and e571's handled list excludes all three, so the 32 cap has no program.
/// ⛔ THE 64 CAP IS NOT A [`sentient::UnrollFactor`], and it does not have to be: only a MEMORY unit
/// takes that arm and a memory op's factor is written to `burst_size`, an integer.
fn round_down(n: u32, ty: DfirUnit, op: &Op, sen_target: SenTarget) -> u32 {
    let Some(n) = NonZeroU32::new(n) else {
        todo!(
            "DT_CHECK_MSG(n != 0, \"Expect a positive target unroll value\") throws: this rerolled \
             statement's unroll size is zero"
        )
    };
    if matches!(
        ty,
        DfirUnit::Lx
            | DfirUnit::L0
            | DfirUnit::Lxlu
            | DfirUnit::Lxsu
            | DfirUnit::L0lu
            | DfirUnit::L0su
    ) {
        n.get().min(64)
    } else {
        round_down_unroll_factor(n, op, sen_target).count()
    }
}

/// `symbolizeSentientUnrollFactor("x" + std::to_string(n)).value()` (`:1063`, `:1148-1151`).
fn unroll_factor_of(n: u32) -> sentient::UnrollFactor {
    match n {
        1 => sentient::UnrollFactor::X1,
        2 => sentient::UnrollFactor::X2,
        3 => sentient::UnrollFactor::X3,
        4 => sentient::UnrollFactor::X4,
        8 => sentient::UnrollFactor::X8,
        _ => todo!(
            "symbolizeSentientUnrollFactor(\"x{n}\").value() is std::nullopt — no SentientUnrollFactor \
             is spelled that"
        ),
    }
}

/// WHICH OF A COMPUTE OP'S NAMED OPERAND BUNDLES — `port_to_op_conversion`'s `std::string name`
/// (`:1122-1145`), which is a fragment of the attribute names that follow it.
///
/// ⛔ NOT [`OperandName`], WHICH IS THE PASS'S OWN PORT-KEYED SLOT. This is the op's `opA`/`opB`/`opC`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpSlot {
    /// `"A"`.
    A,
    /// `"B"`.
    B,
    /// `"C"`.
    C,
}

/// `port_to_op_conversion(port_id, op)` (`:1122-1145`) — which bundle was assigned this compute port.
///
/// ⛔ AN UNASSIGNED BUNDLE NEVER MATCHES, and that is the reference's arithmetic and not a guard added
/// here: `port_id` is `unsigned`, so its `-1` promotes to `4294967295` and compares equal to nothing.
/// ⭐ `None` IS THE REFERENCE'S EMPTY NAME — a unary whose `opAPortID` is not the port asked about
/// (`:1141-1143`). It then sets `"unrollIncrOp"`, `"opDataID"` and `"opPortID"`, which no reader of
/// this dialect names, so an absent bundle writing nothing IS what the reference does.
fn port_to_op_conversion(op: &Op, port_id: i32) -> Option<OpSlot> {
    match op {
        Op::Sentient(sentient::Op::VectorMac { op_a, op_b, .. }) => {
            if op_a.port_id == Some(port_id) {
                Some(OpSlot::A)
            } else if op_b.port_id == Some(port_id) {
                Some(OpSlot::B)
            } else {
                Some(OpSlot::C)
            }
        }
        Op::Sentient(sentient::Op::VectorBinary { op_a, .. }) => {
            if op_a.port_id == Some(port_id) {
                Some(OpSlot::A)
            } else {
                Some(OpSlot::B)
            }
        }
        Op::Sentient(sentient::Op::VectorUnary { op_a, .. }) => {
            (op_a.port_id == Some(port_id)).then_some(OpSlot::A)
        }
        _ => None,
    }
}

/// The operand bundle a slot names, or `None` where this op has no such bundle.
fn operand_mut(op: &mut Op, slot: Option<OpSlot>) -> Option<&mut sentient::Operand> {
    match (op, slot?) {
        (Op::Sentient(sentient::Op::VectorMac { op_a, .. }), OpSlot::A)
        | (Op::Sentient(sentient::Op::VectorBinary { op_a, .. }), OpSlot::A)
        | (Op::Sentient(sentient::Op::VectorUnary { op_a, .. }), OpSlot::A) => Some(op_a),
        (Op::Sentient(sentient::Op::VectorMac { op_b, .. }), OpSlot::B)
        | (Op::Sentient(sentient::Op::VectorBinary { op_b, .. }), OpSlot::B) => Some(op_b),
        (Op::Sentient(sentient::Op::VectorMac { op_c, .. }), OpSlot::C) => Some(op_c),
        _ => None,
    }
}

/// `$ResultForwarding`, `$ResultPrecision` and `$unrollIncrResult` together.
fn result_ports_mut(op: &mut Op) -> Option<&mut sentient::ResultPorts> {
    match op {
        Op::Sentient(
            sentient::Op::VectorMac { result, .. }
            | sentient::Op::VectorBinary { result, .. }
            | sentient::Op::VectorUnary { result, .. },
        ) => Some(result),
        _ => None,
    }
}

/// `$unrollFactor`, which four ops of the dialect carry.
fn unroll_factor_mut(op: &mut Op) -> Option<&mut sentient::UnrollFactor> {
    match op {
        Op::Sentient(
            sentient::Op::VectorMac { unroll_factor, .. }
            | sentient::Op::VectorBinary { unroll_factor, .. }
            | sentient::Op::VectorUnary { unroll_factor, .. }
            | sentient::Op::Splat { unroll_factor, .. },
        ) => Some(unroll_factor),
        _ => None,
    }
}

/// `op->setAttr("burst_size", ...)` — ⭐ AN EXTENT FIELD HERE, which is why `e117_fillMemOpAttrs`
/// has to drop it from the snapshot it compares.
fn burst_size_mut(op: &mut Op) -> Option<&mut Elements> {
    match op {
        Op::Sentient(
            sentient::Op::LoadAndSend { extent, .. } | sentient::Op::ReceiveAndStore { extent, .. },
        ) => Some(&mut extent.burst_size),
        _ => None,
    }
}

/// `getMutableAddrMutable().getOperandNumber()` — the operand the memory chain re-points (`:1234-1242`).
fn mutable_addr_mut(op: &mut Op) -> Option<&mut Val> {
    match op {
        Op::Sentient(
            sentient::Op::LoadAndSend { mutable_addr, .. }
            | sentient::Op::ReceiveAndStore { mutable_addr, .. },
        ) => Some(mutable_addr),
        _ => None,
    }
}

/// `new_mac_op.getPointersMutable()`, IN THE `.td`'S ORDER (`:1315-1320`) — the mac's xrf write
/// pointer then its read pointer, each present only when the op names it.
fn xrf_pointers_mut(op: &mut Op) -> Vec<&mut Val> {
    match op {
        Op::Sentient(sentient::Op::VectorMac {
            xrf_write_ptr,
            xrf_read_ptr,
            ..
        }) => [xrf_write_ptr.as_mut(), xrf_read_ptr.as_mut()]
            .into_iter()
            .flatten()
            .collect(),
        _ => Vec::new(),
    }
}

/// Replaces: e338_setUnrollFieldsInStmt
///
/// Writes the rerolled statement onto its first op — the unroll factor, the per-slot increment flags,
/// the reset data and port ids, the xrf increments — and CLONES that op for every instance the factor
/// could not cover, re-indexing each clone's operands and chaining its address or xrf pointers
/// (`:1038`).
///
/// ⛔ TRAP: A `ref_op` MAY BE ABSENT AND THE FIRST TWO STATEMENTS ARE WHAT SURVIVES IT (`:1041`).
/// ⛔ TRAP: THE `else` BRANCH SETS `unrollIncrOp<B>` FROM THE **C** SLOT — the reference's own
/// `// why this diff? PC: Check with Wei` (`:1173`), preserved.
pub(crate) fn set_unroll_fields_in_stmt(
    block: &mut [Op],
    at: InBlock,
    operand_list: &UnrollOperands,
    ty: DfirUnit,
    closing: &mut Closing<'_>,
) {
    if operand_list.op_name.is_none() || !operand_list.are_unroll_fields_updated {
        return;
    }
    let mut unroll_size = operand_list.unroll_size.0;
    let mut remaining_xrf_write_incr_val = operand_list.xrf_write_incr.0;
    let is_memory_unit = operand_list.is_memory_unit;
    let sen_target = closing.sen_target;

    let mut unroll_factor_val = round_down(unroll_size, ty, &block[at.0], sen_target);
    unroll_size -= unroll_factor_val;
    let mut result_unroll_attr = operand_list.is_field_unroll[OperandName::Result.slot()];

    // Handle splat case first.
    if matches!(&block[at.0], Op::Sentient(sentient::Op::Splat { .. })) {
        if let Op::Sentient(sentient::Op::Splat {
            unroll_incr_result,
            unroll_factor,
            ..
        }) = &mut block[at.0]
        {
            *unroll_incr_result = result_unroll_attr;
            *unroll_factor = unroll_factor_of(unroll_factor_val);
        }
        // If `unroll_size > 0` after resetting the ref_op, more splat operations need to be created.
        let mut clones: Vec<Op> = Vec::new();
        while unroll_size > 0 {
            // Clone the input logical port; the output logical port is recomputed.
            let Some(splat_input) = operand_list.splat_input else {
                todo!(
                    "operand_list.getSplatInput().getDefiningOp() dereferences the null this \
                     snapshot's absent splat input gives"
                )
            };
            let Some(input_op) = defining_op(splat_input, block).cloned() else {
                todo!(
                    "builder.clone(*operand_list.getSplatInput().getDefiningOp()) clones the null \
                     definer of this splat's input"
                )
            };
            let Some(new_input_op) = cloned(&input_op, closing.values) else {
                return;
            };
            let Some(new_output_port) =
                operand_list.create_operand(OperandName::Result, UnrollSize(unroll_size))
            else {
                todo!(
                    "symbolizeSentientComputePort(new_output_port).value() is std::nullopt for the \
                     re-indexed result port of this splat"
                )
            };
            let new_output_op = Op::Sentient(sentient::Op::LogicalPort {
                port_name: new_output_port,
                result: closing.values.mint(),
            });

            // calculate unroll_factor
            unroll_factor_val = round_down(unroll_size, ty, &block[at.0], sen_target);
            unroll_size -= unroll_factor_val;

            // op clone — ⭐ ALWAYS OF THE ORIGINAL: `op` is never reassigned on this path.
            let Some(mut new_op) = cloned(&block[at.0], closing.values) else {
                return;
            };
            result_unroll_attr = unroll_factor_val > 1;
            if let Op::Sentient(sentient::Op::Splat {
                input,
                output,
                unroll_incr_result,
                unroll_factor,
                ..
            }) = &mut new_op
            {
                *unroll_factor = unroll_factor_of(unroll_factor_val);
                *unroll_incr_result = result_unroll_attr;
                // `setOperand(0, ...)` and `setOperand(1, ...)` are `$input` and `$output`.
                if let Some(bound) = results(&new_input_op).first() {
                    *input = *bound;
                }
                if let Some(bound) = results(&new_output_op).first() {
                    *output = *bound;
                }
            }
            clones.extend([new_input_op, new_output_op, new_op]);
        }
        closing.pending.push_after(at, clones);
        return;
    }

    let op_a_unroll_attr = operand_list.is_field_unroll[OperandName::OpA.slot()];
    let op_b_unroll_attr = operand_list.is_field_unroll[OperandName::OpB.slot()];
    let op_c_unroll_attr = operand_list.is_field_unroll[OperandName::OpC.slot()];

    // `op_a`/`op_b`/`op_c` in this pass refer to the operands at port0/port1/port2.
    let mut op_a_name = None;
    let mut op_b_name = None;
    if is_memory_unit {
        if let Some(burst_size) = burst_size_mut(&mut block[at.0]) {
            *burst_size = Elements(u64::from(unroll_factor_val));
        }
    } else {
        if let Some(unroll_factor) = unroll_factor_mut(&mut block[at.0]) {
            *unroll_factor = unroll_factor_of(unroll_factor_val);
        }
        if let Some(result) = result_ports_mut(&mut block[at.0]) {
            result.unroll_incr = result_unroll_attr;
        }
        op_a_name = port_to_op_conversion(&block[at.0], 0);
        if let Some(operand) = operand_mut(&mut block[at.0], op_a_name) {
            operand.unroll_incr = op_a_unroll_attr;
        }

        let binary_mul = match &block[at.0] {
            Op::Sentient(sentient::Op::VectorBinary { binary_op, .. }) => Some(matches!(
                binary_op.op(),
                sentient::BinaryOp::Mul | sentient::BinaryOp::MulDiv2
            )),
            _ => None,
        };
        let (port_id, unroll_attr) = if matches!(ty, DfirUnit::Pe | DfirUnit::Sfp) {
            match binary_mul {
                Some(mul) => {
                    let logical_result =
                        operand_list.is_field_unroll[OperandName::LogicalResult.slot()];
                    // ⭐ ONLY A FORWARDING BINARY HAS A LOGICAL RESULT TO FLAG, and a plain one has
                    // no `unrollIncrLogicalResult` reader either.
                    if let Op::Sentient(sentient::Op::VectorBinary {
                        binary_op: sentient::Binary::Forwarding { unroll_incr, .. },
                        ..
                    }) = &mut block[at.0]
                    {
                        *unroll_incr = logical_result;
                    }
                    if mul {
                        (1, op_b_unroll_attr)
                    } else {
                        // why this diff? PC: Check with Wei
                        (2, op_c_unroll_attr)
                    }
                }
                None => (1, op_b_unroll_attr),
            }
        } else {
            (2, op_b_unroll_attr)
        };
        op_b_name = port_to_op_conversion(&block[at.0], port_id);
        if let Some(operand) = operand_mut(&mut block[at.0], op_b_name) {
            operand.port_id = Some(port_id);
            operand.unroll_incr = unroll_attr;
        }

        // Reset dataId's
        for name in [op_a_name, op_b_name] {
            if let Some(operand) = operand_mut(&mut block[at.0], name) {
                operand.data_id = None;
            }
        }
        // Reset portID's
        if let Some(operand) = operand_mut(&mut block[at.0], op_a_name) {
            operand.port_id = Some(0);
        }

        if matches!(&block[at.0], Op::Sentient(sentient::Op::VectorMac { .. })) {
            let op_c_port = if ty.is_pt_row() { 1 } else { 2 };
            let op_c_name = port_to_op_conversion(&block[at.0], op_c_port);
            if let Some(operand) = operand_mut(&mut block[at.0], op_c_name) {
                operand.unroll_incr = op_c_unroll_attr;
                operand.data_id = None;
                operand.port_id = Some(op_c_port);
            }
            if operand_list.is_xrf_rd_op() {
                let xrf_read_incr_value = if unroll_size == 0 {
                    operand_list.xrf_read_incr.0
                } else {
                    0
                };
                if let Op::Sentient(sentient::Op::VectorMac { xrf_read_incr, .. }) =
                    &mut block[at.0]
                {
                    *xrf_read_incr = xrf_read_incr_value as u32;
                }
            } else if operand_list.is_xrf_wt_op() {
                let xrf_write_incr_value = if unroll_size == 0 {
                    remaining_xrf_write_incr_val
                } else {
                    unroll_factor_val as i32
                };
                remaining_xrf_write_incr_val -= xrf_write_incr_value;
                if let Op::Sentient(sentient::Op::VectorMac { xrf_write_incr, .. }) =
                    &mut block[at.0]
                {
                    *xrf_write_incr = xrf_write_incr_value as u32;
                }
            }
        }
    }
    let _ = (op_a_name, op_b_name);

    // If `unroll_size > 0` after resetting the ref_op, more vector operations need to be created.
    // Cloning is necessary because a sub's operand order has to be preserved.
    let mut source = block[at.0].clone();
    let mut clones: Vec<Op> = Vec::new();
    while unroll_size > 0 {
        if is_memory_unit {
            let Some(mut new_op) = cloned(&source, closing.values) else {
                return;
            };
            // calculate unroll_factor
            unroll_factor_val = round_down(unroll_size, ty, &source, sen_target);
            unroll_size -= unroll_factor_val;
            if let Some(burst_size) = burst_size_mut(&mut new_op) {
                *burst_size = Elements(u64::from(unroll_factor_val));
            }
            // `op->replaceAllUsesWith(new_op)`, then the chain's own read — ⭐ IN THAT ORDER, so the
            // address `new_op` takes from `op` is not the one just redirected.
            let chained = results(&source);
            for (of, with) in chained.iter().zip(results(&new_op)) {
                replace_all_uses_with(block, *of, with);
                replace_all_uses_with(&mut clones, *of, with);
            }
            let Some(address) = chained.first() else {
                todo!(
                    "op->getResult(0) is out of range on a memory op that binds nothing, which is \
                     the reference's own read"
                )
            };
            let Some(mutable_addr) = mutable_addr_mut(&mut new_op) else {
                todo!(
                    "llvm_unreachable(\"Ineligible operation for op rerolling!\") on an op that is \
                     neither a load_and_send nor a receive_and_store"
                )
            };
            *mutable_addr = *address;
            source = new_op.clone();
            clones.push(new_op);
            continue;
        }

        // Recreate the operands and forwarding lists — lrf indices move where the slot is unrolled.
        let new_unroll_size = UnrollSize(unroll_size);
        let slot_of = |port_id: Option<i32>| {
            UnrollOperands::unroll_operand_using_port(ty, ComputePortId::from_port_id(port_id))
        };
        let (a_slot, b_slot, c_slot) = match &source {
            Op::Sentient(sentient::Op::VectorMac {
                op_a, op_b, op_c, ..
            }) => (
                Some(slot_of(op_a.port_id)),
                Some(slot_of(op_b.port_id)),
                Some(slot_of(op_c.port_id)),
            ),
            Op::Sentient(sentient::Op::VectorBinary { op_a, op_b, .. }) => (
                Some(slot_of(op_a.port_id)),
                Some(slot_of(op_b.port_id)),
                None,
            ),
            Op::Sentient(sentient::Op::VectorUnary { op_a, .. }) => {
                (Some(slot_of(op_a.port_id)), None, None)
            }
            _ => (None, None, None),
        };
        let operand_of = |slot: Option<OperandName>| {
            slot.map(|slot| operand_list.create_operand(slot, new_unroll_size))
        };
        let forwarding_of = |slot: Option<OperandName>| {
            slot.map(|slot| operand_list.create_forwarding_array(slot, new_unroll_size))
        };
        let operand_a = operand_of(a_slot);
        let op_a_forwarding = forwarding_of(a_slot);
        let operand_b = operand_of(b_slot);
        let op_b_forwarding = forwarding_of(b_slot);
        let operand_c = operand_of(c_slot);
        let op_c_forwarding = forwarding_of(c_slot);
        let result_forwarding =
            operand_list.create_forwarding_array(OperandName::Result, new_unroll_size);

        // calculate unroll_factor
        unroll_factor_val = round_down(unroll_size, ty, &source, sen_target);
        unroll_size -= unroll_factor_val;

        // op clone
        let Some(mut new_op) = cloned(&source, closing.values) else {
            return;
        };
        if let Some(unroll_factor) = unroll_factor_mut(&mut new_op) {
            *unroll_factor = unroll_factor_of(unroll_factor_val);
        }
        if let Some(result) = result_ports_mut(&mut new_op) {
            result.forwarding = symbolized_array(Some(result_forwarding));
        }

        let xrf_read_incr_val = if operand_list.is_xrf_rd_op() && unroll_size == 0 {
            operand_list.xrf_read_incr.0
        } else {
            0
        };
        let xrf_write_incr_val = if operand_list.is_xrf_wt_op() {
            if unroll_size == 0 {
                remaining_xrf_write_incr_val
            } else {
                unroll_factor_val as i32
            }
        } else {
            0
        };
        remaining_xrf_write_incr_val -= xrf_write_incr_val;

        let is_mac = matches!(&new_op, Op::Sentient(sentient::Op::VectorMac { .. }));
        let is_binary = matches!(&new_op, Op::Sentient(sentient::Op::VectorBinary { .. }));
        let is_unary = matches!(&new_op, Op::Sentient(sentient::Op::VectorUnary { .. }));
        if !is_mac && !is_binary && !is_unary {
            todo!(
                "op->emitError(\"This operation is not supported by opRerolling.\"); \
                 signalPassFailure() — this op reached e338's clone loop"
            )
        }
        let mut bundles = vec![(OpSlot::A, operand_a, op_a_forwarding)];
        if is_mac || is_binary {
            bundles.push((OpSlot::B, operand_b, op_b_forwarding));
        }
        if is_mac {
            bundles.push((OpSlot::C, operand_c, op_c_forwarding));
        }
        for (slot, port, forwarding) in bundles {
            let port = symbolized_port(port);
            let forwarding = symbolized_array(forwarding);
            if let Some(operand) = operand_mut(&mut new_op, Some(slot)) {
                operand.port = port;
                operand.forwarding = forwarding;
            }
        }
        if is_mac && operand_list.is_xrf_op() {
            // reset xrf pointers
            let chained = results(&source);
            for (of, with) in chained.iter().zip(results(&new_op)) {
                replace_all_uses_with(block, *of, with);
                replace_all_uses_with(&mut clones, *of, with);
            }
            for (result_idx, pointer) in xrf_pointers_mut(&mut new_op).into_iter().enumerate() {
                let Some(bound) = chained.get(result_idx) else {
                    todo!(
                        "op->getResult({result_idx}) is out of range on a mac that binds fewer \
                         results than it has xrf pointers, which is the reference's own read"
                    )
                };
                *pointer = *bound;
            }
            if let Op::Sentient(sentient::Op::VectorMac {
                xrf_read_incr,
                xrf_write_incr,
                ..
            }) = &mut new_op
            {
                *xrf_read_incr = xrf_read_incr_val as u32;
                *xrf_write_incr = xrf_write_incr_val as u32;
            }
            source = new_op.clone();
        }
        clones.push(new_op);
    }
    closing.pending.push_after(at, clones);
}

/// `symbolizeSentientComputePort(operand).value()` (`:1306-1314`) over what `e119_createOperand` built.
fn symbolized_port(port: Option<Option<sentient::Port>>) -> sentient::Port {
    match port {
        Some(Some(port)) => port,
        _ => todo!(
            "symbolizeSentientComputePort(operand).value() is std::nullopt for a re-indexed operand \
             of this rerolled statement"
        ),
    }
}

/// The same over what `e120_createForwardingArray` built (`:1315-1317`).
fn symbolized_array(forwarding: Option<Option<Vec<sentient::Port>>>) -> Vec<sentient::Port> {
    match forwarding {
        Some(Some(forwarding)) => forwarding,
        None => Vec::new(),
        Some(None) => todo!(
            "symbolizeSentientComputePort(entry).value() is std::nullopt for a re-indexed forwarding \
             target of this rerolled statement"
        ),
    }
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
pub(crate) fn process_one_block(
    ty: DfirUnit,
    block: &mut Vec<Op>,
    sen_target: SenTarget,
    values: &mut Values,
) {
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
    let mut closing = Closing {
        sen_target,
        values,
        pending: PendingClones::default(),
    };
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
            set_unroll_fields_in_ref_op(block, ref_op_pair.0, &ref_list.0, ty, &mut closing);
            if was_last_fold_mode_ab {
                set_unroll_fields_in_ref_op(block, ref_op_pair.1, &ref_list.1, ty, &mut closing);
                ref_list.0.reset();
                ref_list.1.reset();
            } else {
                ref_list.0.reset();
            }
        }

        // No statement to match against yet: this op — or this pair — starts one (`:317-347`).
        if fold_ab_mode {
            if ref_op_pair.0.is_none() || ref_op_pair.1.is_none() || !was_last_fold_mode_ab {
                ref_list.0.fill(&block[at], block, ty);
                if let Some(second) = curr_op_pair.1 {
                    ref_list.1.fill(&block[second.0], block, ty);
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
            ref_list.0.fill(&block[at], block, ty);
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
        curr_list.0.fill(&block[at], block, ty);
        if fold_ab_mode {
            if let Some(second) = curr_op_pair.1 {
                curr_list.1.fill(&block[second.0], block, ty);
            }
        }

        if was_last_fold_mode_ab {
            let pair_rerolls = fold_ab_mode
                && is_user(block, ref_op_pair.0, curr_op_pair.0, &ops_to_be_deleted)
                && is_user(block, ref_op_pair.1, curr_op_pair.1, &ops_to_be_deleted)
                && ref_list.0.matches(&curr_list.0, block)
                && ref_list.1.matches(&curr_list.1, block)
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
                    &mut closing,
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
                    &mut closing,
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
                    &mut closing,
                );
                close_and_restart(
                    block,
                    ty,
                    &mut ref_op_pair.1,
                    &mut prev_op_pair.1,
                    &mut ref_list.1,
                    curr_op_pair.1,
                    &curr_list.1,
                    &mut closing,
                );
            }
        } else if !fold_ab_mode
            && is_user(block, ref_op_pair.0, curr_op_pair.0, &ops_to_be_deleted)
            && ref_list.0.matches(&curr_list.0, block)
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
                &mut closing,
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
                &mut closing,
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
    // ⭐ ONE PASS FOR BOTH, because e338's clones are held against the position they follow: see
    // [`PendingClones`]. Removing by position and splicing in the same walk keeps every queued index
    // meaning what it meant when it was queued.
    let mut pending = closing.pending;
    let old = core::mem::take(block);
    let mut rebuilt = Vec::with_capacity(old.len());
    for (at, op) in old.into_iter().enumerate() {
        if !doomed.contains(&at) {
            rebuilt.push(op);
        }
        rebuilt.extend(pending.take(InBlock(at)));
    }
    *block = rebuilt;
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
    ref_operand_list: &UnrollOperands,
    ty: DfirUnit,
    closing: &mut Closing<'_>,
) {
    if ref_operand_list.op_name.is_none() || !ref_operand_list.are_unroll_fields_updated {
        return;
    }
    let Some(ref_op) = ref_op else {
        return;
    };
    set_unroll_fields_in_stmt(block, ref_op, ref_operand_list, ty, closing);
}

/// [`update_ref_op_unroll_info`] FOR A PAIR HALF THAT MAY BE ABSENT — the reference passes raw
/// pointers, and both are non-null on every path that reaches the call.
#[allow(clippy::too_many_arguments)]
fn merge_into_ref_op(
    block: &mut [Op],
    ty: DfirUnit,
    curr_op: Option<InBlock>,
    ref_op: Option<InBlock>,
    last_op: InBlock,
    ref_operand_list: &mut UnrollOperands,
    curr_operand_list: &UnrollOperands,
    ops_to_be_deleted: &mut Vec<InBlock>,
    closing: &mut Closing<'_>,
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
        closing,
    );
}

/// THE `else` ARM `processOneBlock` WRITES OUT THREE TIMES (`:365-397`, `:406-419`) — close the
/// statement, then start the next one from `curr_op`'s snapshot, or from nothing where that snapshot
/// is the `"NA"` sentinel.
///
/// ⭐ THIS IS e454 WITH ITS DEAD STORES MADE LIVE, which is exactly what that unit's note says the
/// reference does instead of calling it: `ref_op` and `prev_op` reach the caller here.
#[allow(clippy::too_many_arguments)]
fn close_and_restart(
    block: &mut [Op],
    ty: DfirUnit,
    ref_op: &mut Option<InBlock>,
    prev_op: &mut Option<InBlock>,
    ref_operand_list: &mut UnrollOperands,
    curr_op: Option<InBlock>,
    curr_operand_list: &UnrollOperands,
    closing: &mut Closing<'_>,
) {
    set_unroll_fields_in_ref_op(block, *ref_op, ref_operand_list, ty, closing);
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

/// WHICH XRF POINTER A MAC ADVANCES — `enum XrfType {xrf_read, xrf_write}` (`OpRerolling.hpp:28`).
///
/// ⭐ THE CHOICE IS ALSO A RESULT INDEX: the write pointer is `getResult(0)` and the read pointer
/// `getResult(1)`, at both the pick (`:97-98`) and the rewire (`:121`, `:126`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum XrfType {
    /// `xrf_read`.
    Read,
    /// `xrf_write`.
    Write,
}

impl XrfType {
    /// `(xrf_type == xrf_write) ? mac_op.getResult(0) : mac_op.getResult(1)`.
    const fn result_index(self) -> usize {
        match self {
            XrfType::Write => 0,
            XrfType::Read => 1,
        }
    }
}

/// `mac_op.isXrfRelated()` AND THE DIRECTION IT PICKS (`:91-98`) — `None` for anything that is not an
/// xrf-related mac, so the reference's `if` and its ternary are one question.
///
/// ⭐ THE PREDICATES ARE ALREADY PORTED, at [`MacXrfIncrements`], which is why they are not recomputed
/// here. ⛔ `isXrfWtRelated()` OUTRANKS THE READ SIDE: a mac related both ways takes the write path.
fn mac_xrf_direction(op: &mut Op) -> Option<XrfType> {
    let Op::Sentient(sen_op) = op else {
        return None;
    };
    let incrs = MacXrfIncrements::of(sen_op)?;
    if incrs.wt_related() {
        Some(XrfType::Write)
    } else if incrs.rd_related() {
        Some(XrfType::Read)
    } else {
        None
    }
}

/// `result.hasOneUse() && isa<sentient::AddOp>(*result.getUsers().begin())` (`:100-101`).
///
/// ⛔ A SOLE READER INSIDE A REGION ENDS THE WALK: it is not a top-level position, so the chain stops
/// there rather than reaching into a nested block the erase could not reach either.
fn sole_scalar_add_user(block: &[Op], val: Val) -> Option<usize> {
    if use_count(val, block) != 1 {
        return None;
    }
    block.iter().position(|op| {
        matches!(op, Op::Sentient(sentient::Op::ScalarAdd { .. }))
            && operands(op).contains(&val)
    })
}

/// `isa<sentient::ConstantOp>(parent_op) && parent_op->hasOneUse()` (`:106-109`).
///
/// ⛔ THE REFERENCE ASSERTS WHERE THIS ANSWERS `None`: a bare `isa<>` on the null definer of a block
/// argument is an assertion failure, not a `false`.
fn single_use_scalar_constant(block: &[Op], val: Val) -> Option<usize> {
    let at = block.iter().position(|op| {
        matches!(op, Op::Sentient(sentient::Op::ScalarConstant { result, .. }) if *result == val)
    })?;
    (use_count(val, block) == 1).then_some(at)
}

/// `sentient::ConstantOp::getValue()` for the op binding `val`, `None` where that is not what binds it.
fn constant_increment(block: &[Op], val: Val) -> Option<i64> {
    match defining_op(val, block) {
        Some(Op::Sentient(sentient::Op::ScalarConstant { value, .. })) => Some(*value),
        _ => None,
    }
}

/// `sentient::getIncrementVal(add_op)`'S `AddOp` ARM (`SentientOps.cpp:2085-2096`) — *"assume only one
/// operand is constantOp"*, whose two failures are the reference's own.
fn scalar_add_increment_val(block: &[Op], lhs: Val, rhs: Val) -> i64 {
    match (constant_increment(block, lhs), constant_increment(block, rhs)) {
        (Some(_), Some(_)) => todo!(
            "DT_CHECK(!isa_and_nonnull<sentient::ConstantOp>(add_op.getInp1().getDefiningOp())) \
             throws: both operands of this sentient.scalar_add are constants"
        ),
        (_, Some(value)) | (Some(value), None) => value,
        (None, None) => todo!(
            "DT_ERROR(\"operation does not have an increment value!\") on a sentient.scalar_add \
             whose operands are both non-constant"
        ),
    }
}

/// Replaces: e334_mergeScalarOpIntoMac
///
/// Folds a chain of `sentient.scalar_add`s on a mac's xrf pointer INTO that mac's own increment
/// attribute, rewires the chain's last result onto the mac's pointer result and erases the whole
/// chain with the constants that fed it (`:81`).
///
/// ⛔ TRAP: THE UNUSED `isXrfRdPtrRelated` LAMBDA IS DEAD (`:82-91`) — the walk asks
/// `isXrfRelated()`, which also admits a write-only mac, and nothing calls the lambda.
/// ⛔ TRAP: `getResult(1)` ON A ONE-RESULT MAC IS THE REFERENCE'S OWN OUT-OF-RANGE READ, because
/// `isXrfRdRelated` only asks that the mac bind SOMETHING (`SentientOps.cpp:1656-1665`).
fn merge_scalar_op_into_mac(unit_body: &mut Vec<Op>) {
    // `bb->walk(...)` is post-order: the nested blocks are visited first.
    for at in 0..unit_body.len() {
        for region in regions_mut(&mut unit_body[at]) {
            merge_scalar_op_into_mac(region);
        }
    }
    let mut scalar_op_to_be_deleted: Vec<usize> = Vec::new();
    for at in 0..unit_body.len() {
        let Some(xrf_type) = mac_xrf_direction(&mut unit_body[at]) else {
            continue;
        };
        let Op::Sentient(sentient::Op::VectorMac {
            xrf_read_incr,
            xrf_write_incr,
            ..
        }) = &unit_body[at]
        else {
            continue;
        };
        // `getXrfWriteIncrSigned()` / `getXrfReadIncrSigned()` — `static_cast<int32_t>`.
        let mut accum_xrf_ptr_incr_value = match xrf_type {
            XrfType::Write => *xrf_write_incr as i32,
            XrfType::Read => *xrf_read_incr as i32,
        };
        let Some(mac_result) = results(&unit_body[at])
            .get(xrf_type.result_index())
            .copied()
        else {
            todo!(
                "mac_op.getResult({}) is out of range on an xrf-related mac that binds fewer \
                 results, which is the reference's own read",
                xrf_type.result_index()
            )
        };
        let mut result = mac_result;
        let mut tmp: Vec<usize> = Vec::new();
        while let Some(add_at) = sole_scalar_add_user(unit_body, result) {
            let Op::Sentient(sentient::Op::ScalarAdd {
                lhs,
                rhs,
                result: bound,
                ..
            }) = &unit_body[add_at]
            else {
                break;
            };
            let (lhs, rhs, bound) = (*lhs, *rhs, *bound);
            accum_xrf_ptr_incr_value = accum_xrf_ptr_incr_value
                .wrapping_add(scalar_add_increment_val(unit_body, lhs, rhs) as i32);
            result = bound;
            for operand in [lhs, rhs] {
                if let Some(const_at) = single_use_scalar_constant(unit_body, operand) {
                    tmp.push(const_at);
                }
            }
            tmp.push(add_at);
        }

        // if AddOp is found, update the macOp — `result.getDefiningOp() != mac_op` is exactly *"the
        // walk advanced"*, since every step rebinds `result` to an add's own result.
        if result == mac_result {
            continue;
        }
        if let Op::Sentient(sentient::Op::VectorMac {
            xrf_read_incr,
            xrf_write_incr,
            ..
        }) = &mut unit_body[at]
        {
            match xrf_type {
                XrfType::Read => *xrf_read_incr = accum_xrf_ptr_incr_value as u32,
                XrfType::Write => *xrf_write_incr = accum_xrf_ptr_incr_value as u32,
            }
        }
        replace_all_uses_with(unit_body, result, mac_result);
        scalar_op_to_be_deleted.extend(tmp);
    }
    // `op->dropAllUses(); op->erase();` — every reader of a doomed op is itself doomed, so removing
    // the definition by POSITION is the whole of both, highest-first.
    scalar_op_to_be_deleted.sort_unstable();
    scalar_op_to_be_deleted.dedup();
    for at in scalar_op_to_be_deleted.into_iter().rev() {
        unit_body.remove(at);
    }
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
    sen_target: SenTarget,
    values: &mut Values,
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
            RerollScope::WholeUnit => {
                reroll_blocks(unit_type, &mut unit.body, sen_target, values);
            }
            RerollScope::Op(at) => {
                if let Some(op) = at.op_mut(&mut unit.body) {
                    for region in regions_mut(op) {
                        reroll_blocks(unit_type, region, sen_target, values);
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
fn reroll_blocks(ty: DfirUnit, block: &mut Vec<Op>, sen_target: SenTarget, values: &mut Values) {
    for at in 0..block.len() {
        for region in regions_mut(&mut block[at]) {
            reroll_blocks(ty, region, sen_target, values);
        }
    }
    process_one_block(ty, block, sen_target, values);
}

/// `-dcc-op-rerolling-disable`, `cl::init(false)` (`:54-56`) — a `dcc-opt` command-line flag, not a
/// program property, and this crate has no flags.
const DISABLE_THIS_PASS: bool = false;

/// `opts_.OptLevel == 0` (`:1000`) — the pipeline's optimization level, which this crate compiles at
/// its default and not at `-O0`.
const OPT_LEVEL_ZERO: bool = false;

/// Replaces: e607_runOnOperation
///
/// The pass entry: rerolls every program unit of the module, innermost first, unless the flag turns
/// the pass off or `-O0` finds the unit already fits its instruction buffer (`:994-1006`).
///
/// ⛔ TRAP: THE `-O0` GATE IS A CONJUNCTION AND THE `const` IS ITS LEFT HALF, so `&&` short-circuits
/// and the out-of-scope estimator is never asked — an unconditional `have_ibuff_space` here would
/// `todo!` on every unit of a pipeline the reference runs clean.
pub(crate) fn run_on_operation<A: Arch, M: Model, W: Workload, E: InstructionEstimator>(
    program: &mut Program<A, M, W>,
    ie: &mut E,
    sen_target: SenTarget,
    values: &mut Values,
) {
    if DISABLE_THIS_PASS {
        return;
    }
    for unit in program.units.iter_mut() {
        if OPT_LEVEL_ZERO && ie.have_ibuff_space(&unit.body) {
            continue;
        }
        run_op_rerolling(
            unit,
            &RerollScope::WholeUnit,
            MergeXrfIntoMac::Yes,
            sen_target,
            values,
        );
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Dd2;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::{GroupId, OpIndex, ProgramName, Units};
    use crate::islands::sentient::ProgramUnits;
    use crate::islands::sentient::dialects::sentient::{LrfIndex, Port, Precision, SplatPad};
    use crate::transform::sentient::analyses::OutOfScopeInstructionEstimator;
    use crate::transform::sentient::op_rerolling::unroll_operands::RolledOp;
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

    /// e453 — merging the block's LAST candidate into the statement grows the unroll size, queues that
    /// candidate with the two logical ports only it reads, and then CLOSES the statement onto the
    /// reference op, which is e338 writing the `x2` and the result increment.
    #[test]
    fn e453_merges_the_last_candidate_then_closes_the_statement_on_the_ref_op() {
        let mut block = vec![
            logical_port(50, Port::Lrf(LrfIndex::L0)),
            logical_port(51, Port::Lrf(LrfIndex::L1)),
            splat(50, 51),
            logical_port(52, Port::Lrf(LrfIndex::L2)),
            logical_port(53, Port::Lrf(LrfIndex::L3)),
            splat(52, 53),
        ];
        let mut reference = UnrollOperands::default();
        reference.fill(&block[2].clone(), &block, DfirUnit::Pe);
        let mut candidate = UnrollOperands::default();
        candidate.fill(&block[5].clone(), &block, DfirUnit::Pe);
        let mut to_delete = Vec::new();
        let mut values = Values::default();
        let mut closing = Closing {
            sen_target: SenTarget::Sentient,
            values: &mut values,
            pending: PendingClones::default(),
        };

        update_ref_op_unroll_info(
            &mut block,
            InBlock(5),
            InBlock(2),
            InBlock(5),
            &mut reference,
            &candidate,
            &mut to_delete,
            DfirUnit::Pe,
            &mut closing,
        );

        // ⭐ THE CANDIDATE FIRST, THEN ITS PORTS — and the shared nothing: both of these ports are
        // read by the doomed splat alone.
        assert_eq!(to_delete, vec![InBlock(5), InBlock(3), InBlock(4)]);
        let Op::Sentient(sentient::Op::Splat {
            unroll_factor,
            unroll_incr_result,
            ..
        }) = &block[2]
        else {
            unreachable!("`splat` builds a sentient.splat")
        };
        assert_eq!(*unroll_factor, sentient::UnrollFactor::X2);
        assert!(*unroll_incr_result, "lrf0 -> lrf1 is a field unroll");
        // The statement was closed, so the reference is back at the `"NA"` sentinel.
        assert_eq!(reference.op_name, None);
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
            reg_locale: None,
        })
    }

    /// e519 — TWO SPLATS OFF ONE INPUT PORT BECOME ONE `x2` SPLAT: the first is snapshotted, the
    /// second matches it on the next LRF index, and closing the statement writes the factor and the
    /// result increment onto the survivor.
    ///
    /// ⭐ THE SHARED INPUT PORT SURVIVES and the consumed output port does not — `hasOneUse()` is
    /// asked of each while the doomed splat still reads them both.
    #[test]
    fn e519_rerolls_two_consecutive_splats_into_one_x2_statement() {
        let mut block = vec![
            logical_port(60, Port::West),
            logical_port(50, Port::Lrf(LrfIndex::L0)),
            logical_port(51, Port::Lrf(LrfIndex::L1)),
            splat(60, 50),
            splat(60, 51),
        ];
        let mut values = Values::default();

        process_one_block(DfirUnit::Pe, &mut block, SenTarget::Sentient, &mut values);

        let mut rerolled = splat(60, 50);
        let Op::Sentient(sentient::Op::Splat {
            unroll_factor,
            unroll_incr_result,
            ..
        }) = &mut rerolled
        else {
            unreachable!("`splat` builds a sentient.splat")
        };
        *unroll_factor = sentient::UnrollFactor::X2;
        *unroll_incr_result = true;

        assert_eq!(
            block,
            vec![
                logical_port(60, Port::West),
                logical_port(50, Port::Lrf(LrfIndex::L0)),
                rerolled,
            ]
        );
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
        let mut values = Values::default();
        process_one_block(DfirUnit::Pe, &mut block, SenTarget::Sentient, &mut values);
        assert_eq!(block, untouched);
    }

    /// e454 — the statement is closed onto its ref op and the reference emptied; an absent snapshot is
    /// then NOT copied over it, and a real one is.
    #[test]
    fn e454_closes_the_statement_then_copies_only_a_real_candidate() {
        let mut block = vec![
            scalar_constant(60),
            logical_port(50, Port::Lrf(LrfIndex::L0)),
            splat(60, 50),
        ];
        let mut reference = UnrollOperands::default();
        reference.fill(&block[2].clone(), &block, DfirUnit::Pe);
        reference.are_unroll_fields_updated = true;
        reference.is_field_unroll[OperandName::Result.slot()] = true;
        reference.unroll_size = UnrollSize(2);
        let mut values = Values::default();
        let mut closing = Closing {
            sen_target: SenTarget::Sentient,
            values: &mut values,
            pending: PendingClones::default(),
        };

        update_ref_op_unroll_fields(
            &mut block,
            InBlock(2),
            &mut reference,
            &UnrollOperands::default(),
            DfirUnit::Pe,
            &mut closing,
        );

        let Op::Sentient(sentient::Op::Splat { unroll_factor, .. }) = &block[2] else {
            unreachable!("`splat` builds a sentient.splat")
        };
        assert_eq!(*unroll_factor, sentient::UnrollFactor::X2);
        // ⭐ THE `"NA"` CANDIDATE IS NOT COPIED, so the reference stays at its own reset.
        assert_eq!(reference.op_name, None);

        let mut candidate = UnrollOperands::default();
        candidate.fill(&block[2].clone(), &block, DfirUnit::Pe);
        update_ref_op_unroll_fields(
            &mut block,
            InBlock(2),
            &mut reference,
            &candidate,
            DfirUnit::Pe,
            &mut closing,
        );
        assert_eq!(reference.op_name, Some(RolledOp::Splat));
        assert_eq!(
            reference.operand_list[OperandName::Result.slot()],
            Some(Port::Lrf(LrfIndex::L0))
        );
    }

    /// `%c = sentient.scalar_constant {value = value : si64} : index`.
    fn scalar_constant_of(value: i64, result: u32) -> Op {
        Op::Sentient(sentient::Op::ScalarConstant {
            value,
            result: Val(result),
            reg_locale: sentient::RegType::Imm,
            ty: crate::islands::dataflow_ir::ty::ScalarTy::Index,
            is_symbol: false,
        })
    }

    /// A mac reading the XRF and binding a write and a read pointer — what e334's walk merges a
    /// scalar-add chain into.
    fn xrf_mac() -> Op {
        Op::Sentient(sentient::Op::VectorMac {
            mask: None,
            xrf_write_ptr: None,
            xrf_read_ptr: None,
            results: vec![Val(20), Val(21)],
            op_a: sentient::Operand::from(Port::Lrf(LrfIndex::L0)),
            op_b: sentient::Operand::from(Port::Xrf),
            op_c: sentient::Operand::from(Port::West),
            result: sentient::ResultPorts::default(),
            mode: sentient::FmaMode::FusedMulAdd,
            compute_precision: Precision::Fp16,
            fold_mode: None,
            unroll_factor: sentient::UnrollFactor::X1,
            xrf_read_incr: 0,
            xrf_write_incr: 0,
            data_transfer_only: false,
            is_data_weight: None,
            dbg_name: None,
        })
    }

    /// `%s = sentient.scalar_add %lhs, %rhs`.
    fn scalar_add_of(lhs: u32, rhs: u32, result: u32) -> Op {
        Op::Sentient(sentient::Op::ScalarAdd {
            lhs: Val(lhs),
            rhs: Val(rhs),
            result: Val(result),
            reg: None,
            element_size: None,
            ty: crate::islands::dataflow_ir::ty::ScalarTy::Index,
        })
    }

    /// The PT body e334 folds: a mac binding an XRF read pointer, the constant `3`, and the
    /// `scalar_add` that steps the pointer by it.
    fn pt_xrf_chain() -> Vec<Op> {
        vec![xrf_mac(), scalar_constant_of(3, 22), scalar_add_of(21, 22, 23)]
    }

    /// The mac's read increment and what is left of the block, after the merge.
    fn merged_read_incr(body: &[Op]) -> u32 {
        assert_eq!(body.len(), 1, "the add and its constant were erased");
        let Op::Sentient(sentient::Op::VectorMac { xrf_read_incr, .. }) = &body[0] else {
            unreachable!("`xrf_mac` builds a vector_mac")
        };
        *xrf_read_incr
    }

    /// e338 — a statement wider than any legal factor writes the factor it CAN carry and clones itself
    /// for the rest: a splat of size 5 keeps `x4` and hands the remaining 1 to a clone with its own
    /// input port copy and a result port shifted by the undershoot.
    #[test]
    fn e338_clones_a_splat_for_the_instances_the_factor_cannot_cover() {
        let mut block = vec![
            scalar_constant(60),
            logical_port(50, Port::Lrf(LrfIndex::L0)),
            splat(60, 50),
        ];
        let mut operands = UnrollOperands::default();
        operands.fill(&block[2].clone(), &block, DfirUnit::Pe);
        operands.are_unroll_fields_updated = true;
        operands.is_field_unroll[OperandName::Result.slot()] = true;
        operands.unroll_size = UnrollSize(5);
        let mut values = Values::default();
        let mut closing = Closing {
            sen_target: SenTarget::Sentient,
            values: &mut values,
            pending: PendingClones::default(),
        };

        set_unroll_fields_in_stmt(&mut block, InBlock(2), &operands, DfirUnit::Pe, &mut closing);

        let Op::Sentient(sentient::Op::Splat {
            unroll_factor,
            unroll_incr_result,
            ..
        }) = &block[2]
        else {
            unreachable!("`splat` builds a sentient.splat")
        };
        assert_eq!(*unroll_factor, sentient::UnrollFactor::X4);
        assert!(*unroll_incr_result);

        // ⭐ THE CLONES WAIT AGAINST THE POSITION THEY FOLLOW: three ops for the one instance left.
        let clones = closing.pending.take(InBlock(2));
        assert_eq!(clones.len(), 3);
        let cloned_input = results(&clones[0])[0];
        let Op::Sentient(sentient::Op::LogicalPort {
            port_name,
            result: cloned_output,
        }) = &clones[1]
        else {
            unreachable!("the second clone is the recomputed output port")
        };
        // `lrf0 + (5 - 1)`, the undershoot the surviving `x4` leaves.
        assert_eq!(*port_name, Port::Lrf(LrfIndex::L4));
        let Op::Sentient(sentient::Op::Splat {
            input,
            output,
            unroll_factor,
            unroll_incr_result,
            ..
        }) = &clones[2]
        else {
            unreachable!("the third clone is the splat itself")
        };
        assert_eq!(*input, cloned_input);
        assert_eq!(*output, *cloned_output);
        assert_eq!(*unroll_factor, sentient::UnrollFactor::X1);
        assert!(!*unroll_incr_result, "one instance does not advance");
    }
    /// 571/656 — a PT unit is one of the nine that reroll, so the walk runs over its blocks (this one
    /// closes no statement) and the PT tail then folds the scalar-add chain into the mac's own read
    /// increment, erasing the chain and the constant that fed it.
    #[test]
    fn e571_walks_then_merges_the_scalar_chain_on_the_pt_tail() {
        let mut unit: ProgramUnit<Dd2> = ProgramUnit {
            on: Units::one(DfirUnit::PtRow(Row::checked(0).expect("row 0")), Val(0)),
            precision: None,
            body: pt_xrf_chain(),
            arch: core::marker::PhantomData,
        };
        let mut values = Values::default();

        run_op_rerolling(
            &mut unit,
            &RerollScope::WholeUnit,
            MergeXrfIntoMac::Yes,
            SenTarget::Sentient,
            &mut values,
        );

        assert_eq!(merged_read_incr(&unit.body), 3);
    }

    /// A model, so the program is typed; nothing here reads it.
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

    /// A decode rung, for the same reason.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct AnyRung;
    impl Workload for AnyRung {
        const ROWS: u32 = 1;
        const ACTIVE_CAP: u32 = 64;
    }

    /// 607/656 — the entry walks the module's program units, so its PT unit reaches the same merge
    /// tail e571 does. ⭐ The `-O0` gate short-circuits on the `const`, so the out-of-scope estimator
    /// is handed over and never asked.
    #[test]
    fn e607_rerolls_every_program_unit_of_the_module() {
        let unit: ProgramUnit<Dd2> = ProgramUnit {
            on: Units::one(DfirUnit::PtRow(Row::checked(0).expect("row 0")), Val(0)),
            precision: None,
            body: pt_xrf_chain(),
            arch: core::marker::PhantomData,
        };
        let mut program: Program<Dd2, AnyModel, AnyRung> = Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            preamble: Vec::new(),
            units: ProgramUnits::of(unit, Vec::new()),
            bound: core::marker::PhantomData,
        };
        let mut values = Values::default();
        run_on_operation(
            &mut program,
            &mut OutOfScopeInstructionEstimator,
            SenTarget::Sentient,
            &mut values,
        );

        let unit = program.units.iter().next().expect("the module's one unit");
        assert_eq!(merged_read_incr(&unit.body), 3);
    }
}
