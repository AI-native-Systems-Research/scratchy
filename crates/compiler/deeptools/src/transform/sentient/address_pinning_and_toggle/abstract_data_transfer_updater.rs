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

//! `AddressPinningAndToggle.cpp` — 2 of the campaign's 656 units (dependency level(s) [5, 7]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e592_updateHeadOfChainMutableAddrInitializer` | 592 | 5 | 92 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1799` |
//! | `e638_update` | 638 | 7 | 60 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1738` |


use super::looping_chain_mutable_addr_descriptor::{TransferEnd, mutable_addr_of};
use super::{
    ConditionalConstDataTransferUpdater, DataTransferDescriptor, IntegerSequenceDataTransferUpdater,
    SimpleConstantDataTransferUpdater, ToggleDataTransferUpdater, mutable_addr_mut, op_at,
    set_iter_operand,
};
use crate::bridges::dataflow_ir_to_sentient::vc_vector_operands::OpId;
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::dialects::{self, Definitions, Op, Val, sentient};
use crate::transform::sentient::analyses::{EvaluatedValue, ExpressionEvaluator, OffsetSites};
use crate::transform::sentient::utils::{ConstKind, is_constant};
use crate::transform::sentient::{ForRef, IterArgIndex};

/// `AbstractDataTransferUpdater::getOffset`'s VTABLE (`:1011`, pure virtual — no body, so no unit of
/// its own): the four subclass overrides behind the one dispatch every caller reaches them through.
///
/// ⛔ THE FOUR DISAGREE ON WHAT THEY NEED — only e418 materialises anything — so the shared signature
/// carries the evaluator and the build sites that e011, e012 and e013 ignore.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataTransferUpdater {
    /// `SimpleConstantDataTransferUpdater` (`:1050`).
    SimpleConstant(SimpleConstantDataTransferUpdater),
    /// `ToggleDataTransferUpdater` (`:1082`).
    Toggle(ToggleDataTransferUpdater),
    /// `ConditionalConstDataTransferUpdater` (`:1113`).
    ConditionalConst(ConditionalConstDataTransferUpdater),
    /// `IntegerSequenceDataTransferUpdater` (`:1143`).
    IntegerSequence(IntegerSequenceDataTransferUpdater),
}

impl DataTransferUpdater {
    /// `getOffset(new_immut_addr_ev)` on whichever subclass this is.
    ///
    /// ⛔ NOT IDEMPOTENT FOR [`Self::SimpleConstant`]: e418 BUILDS a value, so two calls leave two ops
    /// — which is why e592 keeps the reference's two calls rather than reusing one offset.
    pub fn get_offset<E: ExpressionEvaluator>(
        self,
        dtd: &DataTransferDescriptor,
        new_immut_addr_ev: EvaluatedValue,
        ty: ScalarTy,
        evaluator: &mut E,
        sites: &mut OffsetSites<'_>,
        walked: &mut Vec<Op>,
    ) -> Val {
        match self {
            Self::SimpleConstant(updater) => {
                updater.get_offset(dtd, new_immut_addr_ev, ty, evaluator, sites, walked)
            }
            Self::Toggle(updater) => updater.get_offset(new_immut_addr_ev),
            Self::ConditionalConst(updater) => updater.get_offset(new_immut_addr_ev),
            Self::IntegerSequence(updater) => updater.get_offset(new_immut_addr_ev),
        }
    }
}

/// Replaces: e592_updateHeadOfChainMutableAddrInitializer
///
/// Adds the pinning offset to the mutable address of a chain's HEAD as EARLY as dominance and the
/// iter arg's own uses allow: at the outermost loop's initializer, else beside that loop, else beside
/// the memory op (`:1799-1890`).
///
/// ⛔ `getOffset` IS CALLED TWICE, AND FOR e418 THAT BUILDS THE OFFSET TWICE (`:1802`, `:1848`) — the
/// second value is only tested for constness and then dropped, and the ops it left stay in the IR.
/// ⛔ EACH INSERT MOVES EVERY LATER SIBLING, so the fall-back arm assigns the memory op's operand
/// BEFORE inserting ahead of it: after the insert `dtd.op`'s recorded path names the new `scalar_add`.
/// ⭐ `DT_CHECK_MSG(cur_val == mutable_addr, ..)` NEEDS NO EXPRESSION: `cur_val` only ever moves in
/// step with `cur_loop`, so no loop found means it is still `mutable_addr_[0]`.
pub(crate) fn update_head_of_chain_mutable_addr_initializer<E: ExpressionEvaluator>(
    unit_body: &mut Vec<Op>,
    dtd: &DataTransferDescriptor,
    end: TransferEnd,
    updater: DataTransferUpdater,
    new_immut_addr_ev: EvaluatedValue,
    ty: ScalarTy,
    evaluator: &mut E,
    sites: &mut OffsetSites<'_>,
) {
    let Some(memory_op) = op_at(&dtd.op, unit_body) else {
        todo!(
            "updateHeadOfChainMutableAddrInitializer: `dtd_.getOperation()` is at {:?}, which this \
             unit body does not reach",
            dtd.op
        )
    };
    let mutable_addr = mutable_addr_of(memory_op, end);
    if defining_of(unit_body, mutable_addr).is_some() {
        // `dtd_.dump(); llvm_unreachable("unhandled case");` (`:1885-1889`).
        todo!(
            "updateHeadOfChainMutableAddrInitializer: llvm_unreachable(\"unhandled case\") — \
             mutable_addr {mutable_addr:?} of the chain head at {:?} is not a block argument \
             (AddressPinningAndToggle.cpp:1885-1889)",
            dtd.op
        )
    }
    let offset = updater.get_offset(dtd, new_immut_addr_ev, ty, evaluator, sites, unit_body);

    // ─── the walk up the chain of iter args (`:1806-1843`) ───
    let mut cur_val = mutable_addr;
    let mut cur_loop: Option<ForRef> = None;
    let mut iter_arg_index: Option<IterArgIndex> = None;
    // `while (auto iter_arg = dyn_cast<BlockArgument>(cur_val))` — an undefined value is one.
    while defining_of(unit_body, cur_val).is_none() {
        let Some((next_loop, arg_number)) = for_arg_of(unit_body, cur_val) else {
            todo!(
                "updateHeadOfChainMutableAddrInitializer: DT_CHECK_MSG(next_loop, \"Expect valid \
                 for op\") — {cur_val:?} is a block argument of something that is not a \
                 `sentient.for` (AddressPinningAndToggle.cpp:1822)"
            )
        };
        // `iter_arg.getArgNumber() - 1` under `DT_CHECK_MSG(next_iter_arg_idx >= 0, ..)`: position 0
        // is the induction variable, which is no iter arg at all.
        let Some(next_iter_arg_idx) = arg_number.checked_sub(1) else {
            todo!(
                "updateHeadOfChainMutableAddrInitializer: DT_CHECK_MSG(next_iter_arg_idx >= 0, \
                 \"expected a valid iter-arg index\") — {cur_val:?} is the induction variable \
                 (AddressPinningAndToggle.cpp:1821)"
            )
        };
        let loop_at = for_op_id_of(unit_body, next_loop);
        let Some(offset_at) = defining_op_id(unit_body, offset) else {
            // `dominates(Value, Operation *)` FOR A BLOCK ARGUMENT dominates its whole loop body, so an
            // offset that is one is the reference's `true` here.
            break;
        };
        if !dominates(&offset_at, &loop_at) {
            break;
        }
        if !can_update_iter_arg(unit_body, cur_val, &dtd.op) {
            break;
        }
        // `cur_loop.getInits()[iter_arg_index]` (`:1841`).
        let init = init_of(unit_body, next_loop, next_iter_arg_idx);
        iter_arg_index = Some(IterArgIndex(next_iter_arg_idx as u32));
        cur_loop = Some(next_loop);
        cur_val = init;
    }

    let both_constant = {
        let regions: [&[Op]; 1] = [unit_body];
        let defs = Definitions::from_innermost(&regions);
        is_constant(cur_val, ConstKind::ScalarConstant, defs)
    };
    // ⛔ THE SECOND `getOffset` (`:1848`) — for e418 a second built value, tested and dropped.
    let second_offset = updater.get_offset(dtd, new_immut_addr_ev, ty, evaluator, sites, unit_body);
    let offset_constant = {
        let regions: [&[Op]; 1] = [unit_body];
        let defs = Definitions::from_innermost(&regions);
        is_constant(second_offset, ConstKind::ScalarConstant, defs)
    };

    match (cur_loop, iter_arg_index) {
        // `new_init = old_init + dtd_.getBaseAddr() - new_immut_addr_ev` (`:1850-1859`).
        (Some(loop_op), Some(index)) if both_constant && offset_constant => {
            let Some(base_addr) = dtd.base_addr() else {
                todo!(
                    "updateHeadOfChainMutableAddrInitializer: DT_CHECK(base_addrs_.size() == 1) \
                     (`:657-660`) — {} base addrs",
                    dtd.base_addrs.len()
                )
            };
            let old_init = evaluator.evaluate_value_handle(cur_val);
            let summed = evaluator.evaluate_sum_handle(old_init, base_addr);
            let new_init = evaluator.evaluate_sub_handle(summed, new_immut_addr_ev);
            // `createOffsetValue(&dtd_.getOperation(), new_init)` (e009) is `buildOffsetValue` plus two
            // builder positions, which the campaign names droppable — the same seam as e418.
            let init_val = evaluator.build_offset_value_of(new_init, sites, unit_body, ty);
            set_iter_operand(unit_body, loop_op, index, init_val);
        }
        // `%add = scalar_add %cur_val, %offset` before the loop, then `for (.. %arg = %add)`
        // (`:1861-1871`).
        (Some(loop_op), Some(index)) => {
            let loop_at = for_op_id_of(unit_body, loop_op);
            let result = sites.values.mint();
            insert_before(unit_body, loop_at.path(), scalar_add(cur_val, offset, result, ty));
            set_iter_operand(unit_body, loop_op, index, result);
        }
        // `%add = scalar_add %mutable_addr, %offset` before the memory op, and the transfer reads it
        // (`:1873-1883`).
        _ => {
            let result = sites.values.mint();
            if let Some(op) = op_at_mut(unit_body, dtd.op.path()) {
                *mutable_addr_mut(op, end) = result;
            }
            insert_before(
                unit_body,
                dtd.op.path(),
                scalar_add(mutable_addr, offset, result, ty),
            );
        }
    }
}

/// `sentient::AddOp::create(builder, loc, type, lhs, rhs)` — the op, with the location the island does
/// not carry and the register no allocator has assigned yet.
fn scalar_add(lhs: Val, rhs: Val, result: Val, ty: ScalarTy) -> Op {
    Op::Sentient(sentient::Op::ScalarAdd {
        lhs,
        rhs,
        result,
        reg: None,
        element_size: None,
        ty,
    })
}

/// `Value::getDefiningOp()` OVER THE WHOLE UNIT — `None` is `isa<BlockArgument>(val)`, since a [`Val`]
/// is bound once ([`crate::islands::dataflow_ir::Values`]).
fn defining_of<'a>(unit_body: &'a [Op], val: Val) -> Option<&'a Op> {
    dialects::defining_op(val, unit_body)
}

/// `dyn_cast<sentient::ForOp>(iter_arg.getOwner()->getParentOp())` WITH `getArgNumber()` — the loop a
/// block argument belongs to, named by its induction variable, and its position among that loop's
/// region arguments.
fn for_arg_of(unit_body: &[Op], val: Val) -> Option<(ForRef, usize)> {
    let (op, arg_number) = dialects::parent_for_arg(val, unit_body)?;
    let Op::Sentient(sentient::Op::For { iv, .. }) = op else {
        return None;
    };
    Some((ForRef(*iv), arg_number))
}

/// `cur_loop.getInits()[i]` — in range by the `DT_CHECK_MSG(next_iter_arg_idx >= 0, ..)` above, since
/// the index came from the loop's own argument list.
fn init_of(unit_body: &[Op], loop_op: ForRef, index: usize) -> Val {
    match dialects::parent_for_arg(loop_op.0, unit_body) {
        Some((Op::Sentient(sentient::Op::For { carried, .. }), _)) => match carried.get(index) {
            Some(entry) => entry.init,
            None => todo!(
                "updateHeadOfChainMutableAddrInitializer: `cur_loop.getInits()[{index}]` of a loop \
                 carrying {} values (AddressPinningAndToggle.cpp:1841)",
                carried.len()
            ),
        },
        _ => todo!(
            "updateHeadOfChainMutableAddrInitializer: {loop_op:?} names no `sentient.for` in this \
             unit body (AddressPinningAndToggle.cpp:1841)"
        ),
    }
}

/// THE POSITION OF THE `sentient.for` named by its induction variable — it is in this body, because
/// the walk only ever names a loop the body itself handed back.
fn for_op_id_of(unit_body: &[Op], loop_op: ForRef) -> OpId {
    match find_op_id(
        unit_body,
        0,
        &|op| matches!(op, Op::Sentient(sentient::Op::For { iv, .. }) if *iv == loop_op.0),
        &mut Vec::new(),
    ) {
        Some(id) => id,
        None => todo!(
            "updateHeadOfChainMutableAddrInitializer: {loop_op:?} names no `sentient.for` in this \
             unit body (AddressPinningAndToggle.cpp:1866)"
        ),
    }
}

/// `val.getDefiningOp()` AS A POSITION.
fn defining_op_id(unit_body: &[Op], val: Val) -> Option<OpId> {
    find_op_id(
        unit_body,
        0,
        &|op| dialects::results(op).contains(&val),
        &mut Vec::new(),
    )
}

/// THE POSITION OF THE FIRST OP SATISFYING `pred`, in the regions-concatenated numbering [`op_at`]
/// reads back — `base` is where the region being walked starts in its owner's flattened child list.
fn find_op_id(
    scope: &[Op],
    base: u32,
    pred: &dyn Fn(&Op) -> bool,
    prefix: &mut Vec<u32>,
) -> Option<OpId> {
    for (index, op) in scope.iter().enumerate() {
        prefix.push(base + index as u32);
        if pred(op) {
            let id = OpId::at(prefix);
            prefix.pop();
            return Some(id);
        }
        let mut offset = 0u32;
        for region in dialects::regions_ref(op) {
            if let Some(found) = find_op_id(region, offset, pred, prefix) {
                prefix.pop();
                return Some(found);
            }
            offset += region.len() as u32;
        }
        prefix.pop();
    }
    None
}

/// `DominanceInfo::dominates(a, b)` FOR TWO POSITIONS — the positional answer
/// (`bridges/dataflow_ir_to_sentient/vc_operand_reuse.rs:349`): an op dominates one in an enclosing
/// block only when it comes first in that block, and dominates everything nested under it.
fn dominates(a: &OpId, b: &OpId) -> bool {
    let (a, b) = (a.path(), b.path());
    match a.iter().zip(b).position(|(la, lb)| la != lb) {
        None => a.len() <= b.len(),
        Some(index) => a.len() == index + 1 && a[index] < b[index],
    }
}

/// `:1826-1840` — whether every user of the iter arg is one this pass may leave behind when it
/// rewrites the arg's initializer: an inner loop, the yield, an op feeding only the yield, or the
/// memory op itself.
fn can_update_iter_arg(unit_body: &[Op], iter_arg: Val, memory_op: &OpId) -> bool {
    let mut acceptable = true;
    for_each_user(unit_body, iter_arg, &mut Vec::new(), 0, &mut |user, at| {
        if !acceptable {
            return;
        }
        let feeds_the_yield = uses_of(unit_body, dialects::results(user).as_slice()) == 1
            && dialects::results(user)
                .iter()
                .find_map(|val| first_user(unit_body, *val))
                .is_some_and(|op| matches!(op, Op::Sentient(sentient::Op::Yield { .. })));
        let is_loop_or_yield = matches!(
            user,
            Op::Sentient(sentient::Op::For { .. } | sentient::Op::Yield { .. })
        );
        if !feeds_the_yield && !is_loop_or_yield && at != *memory_op {
            acceptable = false;
        }
    });
    acceptable
}

/// `val.getUsers()` — every op with `val` among its operands, at any depth, with its position.
fn for_each_user(
    scope: &[Op],
    val: Val,
    prefix: &mut Vec<u32>,
    base: u32,
    visit: &mut impl FnMut(&Op, OpId),
) {
    for (index, op) in scope.iter().enumerate() {
        prefix.push(base + index as u32);
        if dialects::operands(op).contains(&val) {
            visit(op, OpId::at(prefix));
        }
        let mut offset = 0u32;
        for region in dialects::regions_ref(op) {
            for_each_user(region, val, prefix, offset, visit);
            offset += region.len() as u32;
        }
        prefix.pop();
    }
}

/// `Operation::hasOneUse()` — the number of USES of all of an op's results together, an op reading one
/// twice counting twice.
fn uses_of(scope: &[Op], vals: &[Val]) -> usize {
    scope
        .iter()
        .map(|op| {
            let here = dialects::operands(op)
                .iter()
                .filter(|operand| vals.contains(operand))
                .count();
            let nested: usize = dialects::regions_ref(op)
                .into_iter()
                .map(|region| uses_of(region, vals))
                .sum();
            here + nested
        })
        .sum()
}

/// `*val.user_begin()`.
fn first_user<'a>(scope: &'a [Op], val: Val) -> Option<&'a Op> {
    scope.iter().find_map(|op| {
        if dialects::operands(op).contains(&val) {
            return Some(op);
        }
        dialects::regions_ref(op)
            .into_iter()
            .find_map(|region| first_user(region, val))
    })
}

/// THE OP AT A POSITION, MUTABLY — [`op_at`]'s writing half, for the operand the fall-back arm assigns.
fn op_at_mut<'a>(scope: &'a mut Vec<Op>, path: &[u32]) -> Option<&'a mut Op> {
    let (&ordinal, rest) = path.split_first()?;
    if rest.is_empty() {
        return scope.get_mut(ordinal as usize);
    }
    let next = rest[0] as usize;
    let owner = scope.get_mut(ordinal as usize)?;
    let mut base = 0usize;
    for region in dialects::regions_mut(owner) {
        if next < base + region.len() {
            let mut sub: Vec<u32> = rest.to_vec();
            sub[0] = (next - base) as u32;
            return op_at_mut(region, &sub);
        }
        base += region.len();
    }
    None
}

/// `OpBuilder builder(op)` PLUS THE `create` — the new op takes `at`'s slot and `at`'s op moves one
/// later, together with every sibling after it.
fn insert_before(scope: &mut Vec<Op>, path: &[u32], op: Op) {
    let Some((&ordinal, rest)) = path.split_first() else {
        return;
    };
    if rest.is_empty() {
        if (ordinal as usize) <= scope.len() {
            scope.insert(ordinal as usize, op);
        }
        return;
    }
    let next = rest[0] as usize;
    let Some(owner) = scope.get_mut(ordinal as usize) else {
        return;
    };
    let mut base = 0usize;
    for region in dialects::regions_mut(owner) {
        if next < base + region.len() {
            let mut sub: Vec<u32> = rest.to_vec();
            sub[0] = (next - base) as u32;
            insert_before(region, &sub, op);
            return;
        }
        base += region.len();
    }
}

// crustify:todo: e638_update
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1738  (60 body lines, level 7)
//   original  : void AbstractDataTransferUpdater::update()
//   calls     : e007_isPartOfSomeChain, e010_updateVariableOffsetCalculation, e011_getOffset, e012_getOffset, e013_getOffset, e014_dump, e273_isHeadOfChain, e276_updateImmutableAddr, e277_updateVariableOffsetCalculation, e417_isHeadOfLoopingChain, e418_getOffset, e420_updateImmutableAddr, e421_updateConstantMutableAddr, e423_lookup …

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Elements;
    use crate::formats::Bits;
    use crate::islands::dataflow_ir::Values;
    use crate::islands::dataflow_ir::link::SendEnd;
    use crate::islands::sentient::dialects::sentient::{Reg, RegType, ShuffleMode};
    use crate::transform::sentient::address_pinning_and_toggle::{DescriptorMemoryUnit, ToggleSub};
    use crate::transform::sentient::analyses::{OutOfScopeEvaluator, RegionSite};

    /// `%t = sentient.load_and_send` reading its mutable address from `%mutable_addr`.
    fn load_and_send(mutable_addr: Val) -> Op {
        Op::Sentient(sentient::Op::LoadAndSend {
            mutable_addr,
            immutable_addr: Val(102),
            increment: Val(103),
            consumer: SendEnd::to_self(Val(198)),
            result: Val(104),
            extent: sentient::Extent {
                total_elements: Elements(64),
                element_size: Bits(16),
                chunk_size: Elements(1),
                chunk_stride: Elements(1),
                burst_size: Elements(1),
            },
            interleaved_group: Elements(0),
            rotate_val: None,
            dir: None,
            shuffle_mode: ShuffleMode::NoShuffle,
            reg: Reg {
                locale: RegType::Lar,
                index: None,
            },
            dbg_name: None,
        })
    }

    /// `%sub = sentient.scalar_sub` — the toggle's own sub, which is [`DataTransferUpdater::Toggle`]'s
    /// whole offset.
    fn scalar_sub(result: Val) -> Op {
        Op::Sentient(sentient::Op::ScalarSub {
            lhs: Val(105),
            rhs: Val(106),
            result,
            reg: None,
            element_size: None,
            ty: ScalarTy::Index,
        })
    }

    /// `%init = 4096`, then a loop carrying it, whose body holds the chain head reading the carried
    /// argument `%110` — plus `offset_before_the_loop`'s sub either above the loop or inside it.
    fn chain(offset_inside_the_loop: bool) -> Vec<Op> {
        let sub = scalar_sub(Val(120));
        let mut body = vec![
            load_and_send(Val(110)),
            Op::Sentient(sentient::Op::Yield {
                results: vec![Val(110)],
            }),
        ];
        let mut unit = vec![Op::Sentient(sentient::Op::ScalarConstant {
            value: 4096,
            result: Val(101),
            reg_locale: RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })];
        if offset_inside_the_loop {
            body.insert(0, sub);
        } else {
            unit.push(sub);
        }
        unit.push(Op::Sentient(sentient::Op::For {
            iv: Val(107),
            bound: Val(108),
            bound_reg: None,
            carried: vec![sentient::Carried {
                init: Val(101),
                arg: Val(110),
                result: Val(111),
                reg: Reg {
                    locale: RegType::Lbr,
                    index: None,
                },
                program_header: false,
                element_size: None,
            }],
            dbg_name: None,
            body,
        }));
        unit
    }

    /// The transfer at `at`, with no pattern and one base address.
    fn transfer(at: &[u32]) -> DataTransferDescriptor {
        DataTransferDescriptor {
            op: OpId::at(at),
            pattern_desc: None,
            base_addrs: vec![EvaluatedValue(7)],
            region: RegionSite::ProgramUnitBody,
            memory_unit: DescriptorMemoryUnit::Lx,
            base_addr: Val(0),
            is_base_addr_mutable: false,
        }
    }

    /// The loop's carried initializer, and the mutable address the chain head reads.
    fn init_and_mutable_addr(unit: &[Op], loop_index: usize, head_index: usize) -> (Val, Val) {
        let Op::Sentient(sentient::Op::For { carried, body, .. }) = &unit[loop_index] else {
            panic!("the loop survives")
        };
        (
            carried[0].init,
            mutable_addr_of(&body[head_index], TransferEnd::Src),
        )
    }

    /// 592/656 — an offset defined ABOVE the loop reaches the initializer: the `scalar_add` lands
    /// before the loop and the loop carries its result. An offset defined INSIDE the loop does not
    /// dominate it, so the add lands beside the transfer instead and the transfer reads it.
    #[test]
    fn e592_adds_the_offset_at_the_loop_initializer_unless_it_cannot_dominate_the_loop() {
        let updater = DataTransferUpdater::Toggle(ToggleDataTransferUpdater {
            toggle_sub: ToggleSub::of(Val(120)),
        });
        let mut consts = Vec::new();
        let mut values = Values::default();
        let mut sites = OffsetSites {
            consts: &mut consts,
            query_maps: None,
            values: &mut values,
        };

        // The sub sits at `[1]`, the loop at `[2]`, the chain head at `[2, 0]`.
        let mut hoisted = chain(false);
        update_head_of_chain_mutable_addr_initializer(
            &mut hoisted,
            &transfer(&[2, 0]),
            TransferEnd::Src,
            updater,
            EvaluatedValue(9),
            ScalarTy::Index,
            &mut OutOfScopeEvaluator,
            &mut sites,
        );
        assert_eq!(hoisted.len(), 4);
        assert!(matches!(
            hoisted[2],
            Op::Sentient(sentient::Op::ScalarAdd {
                lhs: Val(101),
                rhs: Val(120),
                result: Val(0),
                ..
            })
        ));
        assert_eq!(init_and_mutable_addr(&hoisted, 3, 0), (Val(0), Val(110)));

        // The sub is now at `[1, 0]`, inside the loop at `[1]`, with the head at `[1, 1]`.
        let mut nested = chain(true);
        update_head_of_chain_mutable_addr_initializer(
            &mut nested,
            &transfer(&[1, 1]),
            TransferEnd::Src,
            updater,
            EvaluatedValue(9),
            ScalarTy::Index,
            &mut OutOfScopeEvaluator,
            &mut sites,
        );
        assert_eq!(nested.len(), 2);
        assert!(matches!(
            nested[1],
            Op::Sentient(sentient::Op::For { ref body, .. })
                if matches!(
                    body[1],
                    Op::Sentient(sentient::Op::ScalarAdd {
                        lhs: Val(110),
                        rhs: Val(120),
                        result: Val(1),
                        ..
                    })
                )
        ));
        assert_eq!(init_and_mutable_addr(&nested, 1, 2), (Val(101), Val(1)));
    }
}
