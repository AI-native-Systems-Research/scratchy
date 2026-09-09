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

//! `AddressPinningAndToggle.cpp` — 3 of the campaign's 656 units (dependency level(s) [0, 1, 2]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e018_getOutermostConstInitAndLoopingChainIncrement` | 018 | 0 | 168 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:3134` |
//! | `e282_LoopingChainMutableAddrDescriptor` | 282 | 1 | 27 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:3081` |
//! | `e427_dump` | 427 | 2 | 12 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:3121` |

// crustify:todo: e427_dump
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:3121  (12 body lines, level 2)
//   original  : void LoopingChainMutableAddrDescriptor::dump() const
//   calls     : e278_isValid, e279_canBeSimplified

use crate::islands::sentient::dialects::{
    Definitions, Op, Val, operands, regions_ref, sentient, use_count,
};
use crate::transform::sentient::analyses::{EvaluatedValue, ExpressionEvaluator, ScalarOffset};
use crate::transform::sentient::utils::{ConstKind, is_constant};
use crate::transform::sentient::{ForRef, IterArgIndex};

/// HOW LONG A LOOPING CHAIN IS — `unsigned size_`, where `isValid()` requires `>= 1` (`:562-564`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ChainSize(pub u32);

/// THE HEAD OF A LOOPING CHAIN OF MEMORY OPS — `class LoopingChainMutableAddrDescriptor`
/// (`AddressPinningAndToggle.cpp:544-629`). Applies to a MUTABLE address only, and carries the
/// chain's total increment so the rest of the chain's transfers can be ignored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LoopingChainMutableAddrDescriptor {
    /// `is_head_of_chain_` — set once the pattern is analysed.
    pub is_head_of_chain: bool,
    /// `outer_loop_` — the outermost loop where `%argN` is initialised.
    pub outer_loop: Option<ForRef>,
    /// `iter_arg_index_` — which iter arg of `outer_loop` starts the chain.
    pub iter_arg_index: Option<IterArgIndex>,
    /// `size_` — the size of the looping chain.
    pub size: ChainSize,
    /// `init_` — the initial value of the mutable addresses in the chain.
    pub init: Option<EvaluatedValue>,
    /// `increment_` — the chain's total increment.
    pub increment: Option<EvaluatedValue>,
    /// `can_be_simplified_`, the base class's own field (`AddressPinningAndToggle.cpp:159`) — true
    /// when the chain is one transfer long or never moves, so it is really one address.
    pub can_be_simplified: bool,
}

/// WHICH END OF A TRANSFER A DESCRIPTOR IS ABOUT — `int mutable_addr_result_idx_` (`:613`), which
/// `getMutableAndImmutableAddr` requires to be 0 or 1 (`Analyses/Utils.cpp:577-578`) and
/// `getIncrementVal` reads as `target_mem_unit_is_src` (`SentientOps.cpp:2073`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferEnd {
    /// `result_idx == 0` — the source half, and the only index a one-result memory op has.
    Src,
    /// `result_idx == 1` — the destination half of a `sentient.load_and_store`.
    Dst,
}

/// WHAT THE CHAIN WALK ANSWERS — the reference's `tuple<ForOp, IndexTy, IndexTy,
/// const EvaluatedValue *>` (`:596-597`).
///
/// ⛔ WHOLE-`None` IS ITS `{nullptr, -1, -1, nullptr}`; `iter_arg_index: None` is the walk that
/// reached a loop but no constant initialiser, which the caller invalidates on (`:3100`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutermostConstInitAndChainIncrement {
    /// The outermost loop the walk reached, named by its induction variable.
    pub outer_loop: ForRef,
    /// Which iter arg of it a constant initialises.
    pub iter_arg_index: Option<IterArgIndex>,
    /// The unrolled size of the looping chain.
    pub size: ChainSize,
    /// The chain's total increment.
    pub increment: EvaluatedValue,
}

impl LoopingChainMutableAddrDescriptor {
    /// Replaces: e282_LoopingChainMutableAddrDescriptor
    ///
    /// Matches `base_addr` as the head of a looping chain of transfers, recording the outermost loop,
    /// its index, the chain's unrolled size, its initial value and its total increment (`:3081-3118`).
    ///
    /// ⛔ A NON-BLOCK-ARGUMENT `base_addr` IS AN ABORT HERE, not a quiet default (`:3096`) — this is
    /// the one descriptor whose constructor asserts rather than returning unmatched.
    /// ⭐ `op_` AND `mutable_addr_result_idx_` BECOME THE `end` ARGUMENT: the index is the only one of
    /// the two the walk reads, and `op_` is read by `e427_dump` alone.
    #[must_use]
    pub fn new(
        base_addr: Val,
        end: TransferEnd,
        scope: &[Op],
        defs: Definitions<'_>,
        evaluator: &mut impl ExpressionEvaluator,
    ) -> LoopingChainMutableAddrDescriptor {
        // `DT_CHECK_MSG(iter_arg, "Expected block argument forming the looping chain\n")` (`:3096`).
        if defs.for_arg_of(base_addr).is_none() {
            todo!(
                "LoopingChainMutableAddrDescriptor: DT_CHECK_MSG(iter_arg, \"Expected block \
                 argument forming the looping chain\") — {base_addr:?} is no loop region argument \
                 (:3096)"
            )
        }

        let found =
            LoopingChainMutableAddrDescriptor::outermost_const_init_and_looping_chain_increment(
                base_addr, end, scope, defs, evaluator,
            );
        let mut desc = match found {
            None => LoopingChainMutableAddrDescriptor::default(),
            Some(found) => LoopingChainMutableAddrDescriptor {
                is_head_of_chain: false,
                outer_loop: Some(found.outer_loop),
                iter_arg_index: found.iter_arg_index,
                size: found.size,
                init: None,
                increment: Some(found.increment),
                can_be_simplified: false,
            },
        };

        // `if (!outer_loop_ || iter_arg_index_ < 0) { invalidate(); return; }` (`:3100-3106`).
        let (Some(outer_loop), Some(iter_arg_index)) = (desc.outer_loop, desc.iter_arg_index)
        else {
            desc.invalidate();
            return desc;
        };
        // `Value curr_init = outer_loop_.getIterOperands()[iter_arg_index_]` (`:3107`).
        let Some(curr_init) = super::iter_operand_of(outer_loop, iter_arg_index, defs) else {
            desc.invalidate();
            return desc;
        };
        if !is_constant(curr_init, ConstKind::ScalarConstant, defs) {
            desc.invalidate();
            return desc;
        }
        desc.init = Some(evaluator.evaluate_value_handle(curr_init));
        desc.is_head_of_chain = true;

        let zero = evaluator.constant(0);
        desc.can_be_simplified = desc.size == ChainSize(1) || desc.increment == Some(zero);
        desc
    }

    /// Replaces: e018_getOutermostConstInitAndLoopingChainIncrement
    ///
    /// Walks the iter-arg chain inner to outer, taking each level's stride from either a
    /// `sentient.scalar_add` or a chain of memory ops' increments, and folds it into
    /// `(increment + stride) * bound` (`:3134-3303`).
    ///
    /// ⛔ A NEGATIVE STRIDE IS REFUSED, NOT NEGATED (`:3260`), and so is a non-constant or
    /// non-positive bound — every refusal is the reference's all-null tuple.
    /// ⛔ THE TAIL WALK IS SKIPPED WHEN NO CONSTANT INIT WAS FOUND: `outer_loop.getResult(-1)`
    /// (`:3282`) reads out of range in the reference, and the caller invalidates on that index.
    #[must_use]
    pub fn outermost_const_init_and_looping_chain_increment(
        iter_arg: Val,
        end: TransferEnd,
        scope: &[Op],
        defs: Definitions<'_>,
        evaluator: &mut impl ExpressionEvaluator,
    ) -> Option<OutermostConstInitAndChainIncrement> {
        let mut iter_arg_index: Option<IterArgIndex> = None;
        let mut curr: Option<Val> = Some(iter_arg);
        let mut outer_loop: Option<&Op> = None;
        let mut prev: Option<(&Op, usize)> = None;
        let mut increment = evaluator.constant(0);
        let mut size: u32 = 0;

        // `while (iter_arg_index < 0 && cur_iter_arg)` (`:3166`)
        while iter_arg_index.is_none() {
            let Some(curr_arg) = curr else { break };
            let (for_op, arg_number) = defs.for_arg_of(curr_arg)?;
            let curr_it_index = arg_number.checked_sub(1)?;
            outer_loop = Some(for_op);
            let Op::Sentient(sentient::Op::For {
                bound,
                carried,
                body,
                ..
            }) = for_op
            else {
                return None;
            };
            let Some(Op::Sentient(sentient::Op::ScalarConstant {
                value: bound_value, ..
            })) = defs.of(*bound)
            else {
                return None;
            };
            if *bound_value <= 0 {
                return None;
            }

            let Some(Op::Sentient(sentient::Op::Yield { results: yielded })) = body.last() else {
                todo!(
                    "getOutermostConstInitAndLoopingChainIncrement: DT_CHECK_MSG(yield, \
                     \"expected terminator of loop body to be a yield\") (:3193)"
                )
            };
            let yield_opnd = *yielded.get(curr_it_index)?;

            // The stride comes from EITHER a `scalar_add` whose left summand is the previous
            // loop's result, OR a chain of memory ops walked back to `cur_iter_arg` (`:3189-3252`).
            // ⭐ `LoadAndExtractScalarOp` IS DELIBERATELY ABSENT — the reference's own note at
            // `:3196` is "not supported outside LX".
            let stride = match defs.of(yield_opnd) {
                Some(Op::Sentient(sentient::Op::ScalarAdd { lhs, rhs, .. })) => {
                    let (prev_loop, prev_it_index) = prev?;
                    if carried_result(prev_loop, prev_it_index) != Some(*lhs)
                        || !is_constant(*rhs, ConstKind::ScalarConstant, defs)
                    {
                        return None;
                    }
                    evaluator.evaluate_value_handle(*rhs)
                }
                Some(
                    tail @ Op::Sentient(
                        sentient::Op::LoadAndSend { .. }
                        | sentient::Op::ReceiveAndStore { .. }
                        | sentient::Op::LoadAndStore { .. },
                    ),
                ) => {
                    let mut cur_op = tail;
                    let mut stride = evaluator.constant(0);
                    loop {
                        size = size.saturating_add(1);
                        let mutable_addr = mutable_addr_of(cur_op, end);
                        let incr_field = increment_val(cur_op, end, defs);
                        let incr_ev = evaluator.constant(incr_field);
                        stride = evaluator.evaluate_sum_handle(stride, incr_ev);
                        if mutable_addr == curr_arg {
                            break;
                        }
                        match defs.of(mutable_addr) {
                            Some(
                                next @ Op::Sentient(
                                    sentient::Op::LoadAndSend { .. }
                                    | sentient::Op::ReceiveAndStore { .. }
                                    | sentient::Op::LoadAndStore { .. },
                                ),
                            ) => cur_op = next,
                            _ if prev.is_some_and(|(prev_loop, prev_it_index)| {
                                carried_result(prev_loop, prev_it_index) == Some(mutable_addr)
                            }) =>
                            {
                                break;
                            }
                            _ => return None,
                        }
                    }
                    stride
                }
                // `if (!match_found)` — a block argument, a non-memory op, or nothing (`:3253`).
                _ => return None,
            };

            if evaluator.is_any_val_less_than(stride, ScalarOffset(0)) {
                return None;
            }
            // `increment = (increment + stride) * bound; size *= bound;` (`:3267-3270`) — ⛔ the
            // reference's `unsigned size` WRAPS here; saturation cannot make a chain look shorter.
            let sum = evaluator.evaluate_sum_handle(increment, stride);
            increment = evaluator.evaluate_multiply_by_const_of(sum, *bound_value);
            size = size.saturating_mul(u32::try_from(*bound_value).unwrap_or(u32::MAX));

            let curr_init = carried.get(curr_it_index)?.init;
            if defs.for_arg_of(curr_init).is_some() {
                curr = Some(curr_init);
                prev = Some((for_op, curr_it_index));
            } else {
                if is_constant(curr_init, ConstKind::ScalarConstant, defs) {
                    iter_arg_index = u32::try_from(curr_it_index).ok().map(IterArgIndex);
                }
                curr = None;
            }
        }

        let outer_loop = outer_loop?;
        // `if (outer_loop_result.hasOneUse())` and the walk over the chain's tail (`:3281-3300`).
        if let Some(index) = iter_arg_index {
            let result = carried_result(outer_loop, index.0 as usize)?;
            if use_count(result, scope) == 1 {
                let mut cur = first_user(result, scope);
                while let Some(
                    cur_op @ Op::Sentient(
                        sentient::Op::LoadAndSend { .. }
                        | sentient::Op::ReceiveAndStore { .. }
                        | sentient::Op::LoadAndStore { .. },
                    ),
                ) = cur
                {
                    let incr_field = increment_val(cur_op, end, defs);
                    let incr_ev = evaluator.constant(incr_field);
                    increment = evaluator.evaluate_sum_handle(increment, incr_ev);
                    let cur_result = mutable_result(cur_op, end)?;
                    match use_count(cur_result, scope) {
                        0 => break,
                        1 => cur = first_user(cur_result, scope),
                        uses => todo!(
                            "getOutermostConstInitAndLoopingChainIncrement: \
                             DT_CHECK_MSG(cur_result.hasOneUse(), \"Expect at most one use of a \
                             memory op result in the chain.\") — {uses} uses (:3295-3297)"
                        ),
                    }
                }
            }
        }

        let Op::Sentient(sentient::Op::For { iv, .. }) = outer_loop else {
            return None;
        };
        Some(OutermostConstInitAndChainIncrement {
            outer_loop: ForRef(*iv),
            iter_arg_index,
            size: ChainSize(size),
            increment,
        })
    }
}

/// `loop.getResult(index)` for a `sentient.for` — the result bound to carried value `index`.
fn carried_result(loop_op: &Op, index: usize) -> Option<Val> {
    let Op::Sentient(sentient::Op::For { carried, .. }) = loop_op else {
        return None;
    };
    Some(carried.get(index)?.result)
}

/// The MUTABLE half of `dcc::utils::getMutableAndImmutableAddr(op, result_idx)`
/// (`Analyses/Utils.cpp:569-605`); e018 `std::ignore`s the immutable half (`:3219-3221`).
///
/// ⛔ THE REFERENCE'S `LoadAndExtractScalarOp` ARM DEREFERENCES THE FAILED `dyn_cast`
/// (`Analyses/Utils.cpp:590`, `receive_and_store.getMutableAddr()`); e018's own `isa` filter
/// (`:3207-3208`, `:3232-3234`) never reaches it, so only the three ops it admits are written here.
fn mutable_addr_of(op: &Op, end: TransferEnd) -> Val {
    match op {
        Op::Sentient(
            sentient::Op::LoadAndSend { mutable_addr, .. }
            | sentient::Op::ReceiveAndStore { mutable_addr, .. },
        ) => *mutable_addr,
        Op::Sentient(sentient::Op::LoadAndStore {
            src_mutable_addr,
            dst_mutable_addr,
            ..
        }) => match end {
            TransferEnd::Src => *src_mutable_addr,
            TransferEnd::Dst => *dst_mutable_addr,
        },
        _ => todo!(
            "getMutableAndImmutableAddr: DT_CHECK(isa<LoadAndSendOp, ReceiveAndStoreOp, \
             LoadAndStoreOp, LoadAndExtractScalarOp, LoadComputeAndSendOp>(op)) on {op:?} \
             (Analyses/Utils.cpp:571-576)"
        ),
    }
}

/// `sentient::getIncrementVal(op, target_mem_unit_is_src)` (`SentientOps.cpp:2065-2101`) for the
/// three ops a looping chain admits. Both stops are the reference's own aborts: its `cast` to
/// `ConstantOp` (`:2067`) and its trailing `DT_ERROR` (`:2099`).
fn increment_val(op: &Op, end: TransferEnd, defs: Definitions<'_>) -> i64 {
    let increment = match op {
        Op::Sentient(
            sentient::Op::LoadAndSend { increment, .. }
            | sentient::Op::ReceiveAndStore { increment, .. },
        ) => *increment,
        Op::Sentient(sentient::Op::LoadAndStore {
            src_inc, dst_inc, ..
        }) => match end {
            TransferEnd::Src => *src_inc,
            TransferEnd::Dst => *dst_inc,
        },
        _ => todo!(
            "getIncrementVal: DT_ERROR(\"operation does not have an increment value!\") on {op:?} \
             (SentientOps.cpp:2099)"
        ),
    };
    match defs.of(increment) {
        Some(Op::Sentient(sentient::Op::ScalarConstant { value, .. })) => *value,
        _ => todo!(
            "getIncrementVal: cast<sentient::ConstantOp>({increment:?}.getDefiningOp()) \
             (SentientOps.cpp:2067)"
        ),
    }
}

/// `op->getResult(mutable_addr_result_idx_)` (`:3293`) — ⛔ `None` for the second result of a
/// one-result memory op, which the reference reads out of range.
fn mutable_result(op: &Op, end: TransferEnd) -> Option<Val> {
    match op {
        Op::Sentient(
            sentient::Op::LoadAndSend { result, .. } | sentient::Op::ReceiveAndStore { result, .. },
        ) => matches!(end, TransferEnd::Src).then_some(*result),
        Op::Sentient(sentient::Op::LoadAndStore { results, .. }) => Some(match end {
            TransferEnd::Src => results.0,
            TransferEnd::Dst => results.1,
        }),
        _ => None,
    }
}

/// `*val.user_begin()` — the op that reads `val`, searched through nested regions. ⭐ ONLY CALLED
/// WHERE THE USE COUNT IS ONE, so the reference's unordered use list cannot disagree with this order.
fn first_user(val: Val, scope: &[Op]) -> Option<&Op> {
    for op in scope {
        if operands(op).contains(&val) {
            return Some(op);
        }
        for region in regions_ref(op) {
            if let Some(user) = first_user(val, region) {
                return Some(user);
            }
        }
    }
    None
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Elements;
    use crate::formats::Bits;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::sentient::{
        Carried, Extent, Reg, RegType, ShuffleMode,
    };
    use crate::transform::sentient::analyses::{Evaluation, MinMax, OffsetSites};

    /// The handle flavour with its answers stated as INTEGERS: one entry per interned value, so the
    /// chain's `(increment + stride) * bound` is observable.
    #[derive(Default)]
    struct StatedEvaluator {
        held: Vec<i64>,
        constants: Vec<(Val, i64)>,
    }

    impl StatedEvaluator {
        fn intern(&mut self, value: i64) -> EvaluatedValue {
            let index = self
                .held
                .iter()
                .position(|held| *held == value)
                .unwrap_or_else(|| {
                    self.held.push(value);
                    self.held.len() - 1
                });
            EvaluatedValue(u32::try_from(index).unwrap_or_default())
        }

        fn value(&self, ev: EvaluatedValue) -> i64 {
            self.held
                .get(usize::try_from(ev.0).unwrap_or_default())
                .copied()
                .unwrap_or_default()
        }
    }

    impl ExpressionEvaluator for StatedEvaluator {
        fn evaluate_value(&mut self, _value: Val) -> Evaluation {
            todo!("e018 asks for handles, never for a decoded evaluation")
        }

        fn evaluate_sum(&mut self, _lhs: &Evaluation, _rhs: &Evaluation) -> Evaluation {
            todo!("e018 asks for handles, never for a decoded evaluation")
        }

        fn build_offset_value(
            &mut self,
            _evaluation: &Evaluation,
            _sites: &mut OffsetSites<'_>,
            _walked: &mut Vec<Op>,
            _ty: ScalarTy,
        ) -> Val {
            todo!("e018 builds no value")
        }

        fn evaluate_value_handle(&mut self, value: Val) -> EvaluatedValue {
            let held = self
                .constants
                .iter()
                .find(|(val, _)| *val == value)
                .map(|(_, held)| *held);
            match held {
                Some(held) => self.intern(held),
                None => todo!("the fixture states no constant for {value:?}"),
            }
        }

        fn constant(&mut self, value: i64) -> EvaluatedValue {
            self.intern(value)
        }

        fn evaluate_sum_handle(
            &mut self,
            lhs: EvaluatedValue,
            rhs: EvaluatedValue,
        ) -> EvaluatedValue {
            let sum = self.value(lhs) + self.value(rhs);
            self.intern(sum)
        }

        fn evaluate_multiply_by_const_of(&mut self, ev: EvaluatedValue, by: i64) -> EvaluatedValue {
            let product = self.value(ev) * by;
            self.intern(product)
        }

        fn evaluate_min_max(&mut self, values: &[EvaluatedValue], which: MinMax) -> EvaluatedValue {
            let mut held: Vec<i64> = values.iter().map(|ev| self.value(*ev)).collect();
            held.sort_unstable();
            let picked = match which {
                MinMax::Min => held.first().copied(),
                MinMax::Max => held.last().copied(),
            };
            self.intern(picked.unwrap_or_default())
        }

        fn is_any_val_less_than(&mut self, ev: EvaluatedValue, bound: ScalarOffset) -> bool {
            self.value(ev) < bound.0
        }
    }

    fn constant(value: i64, result: Val) -> Op {
        Op::Sentient(sentient::Op::ScalarConstant {
            value,
            result,
            reg_locale: RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    fn carried(init: Val, arg: Val, result: Val) -> Carried {
        Carried {
            result_reg: Reg::UNALLOCATED,
            init,
            arg,
            result,
            reg: Reg {
                locale: RegType::Lar,
                index: None,
            },
            program_header: false,
            element_size: None,
        }
    }

    fn load_and_store(src_mutable_addr: Val, src_inc: Val, results: (Val, Val)) -> Op {
        Op::Sentient(sentient::Op::LoadAndStore {
            src: Val(90),
            dst: Val(91),
            src_mutable_addr,
            src_immutable_addr: Val(92),
            src_inc,
            dst_mutable_addr: Val(94),
            dst_immutable_addr: Val(95),
            dst_inc: Val(5),
            multicast_info: None,
            results,
            extent: Extent::of(Elements(8), Bits(16)),
            stride: 1,
            rotate_val: None,
            shuffle_mode: ShuffleMode::NoShuffle,
            src_reg: Reg {
                locale: RegType::Lar,
                index: None,
            },
            dst_reg: Reg {
                locale: RegType::Lbr,
                index: None,
            },
            dir: None,
            is_ibr_write: false,
            dbg_name: None,
        })
    }

    /// The reference's own example (`:536-542`) with the tail of `:3153` attached: an inner loop whose
    /// transfer advances its own address by 2, an outer `scalar_add` of 5, and one more transfer
    /// outside the nest advancing by 7. `inner_bound` is the inner loop's trip count.
    fn looping_chain(inner_bound: Val) -> Vec<Op> {
        let inner = Op::Sentient(sentient::Op::For {
            iv_reg: sentient::Reg::UNALLOCATED,
            iv: Val(20),
            bound: inner_bound,
            carried: vec![carried(Val(11), Val(21), Val(22))],
            dbg_name: None,
            body: vec![
                load_and_store(Val(21), Val(5), (Val(30), Val(31))),
                Op::Sentient(sentient::Op::Yield {
                    results: vec![Val(30)],
                }),
            ],
        });
        vec![
            constant(3, Val(1)),
            constant(4, Val(2)),
            constant(0, Val(3)),
            constant(5, Val(4)),
            constant(2, Val(5)),
            constant(7, Val(6)),
            Op::Sentient(sentient::Op::For {
                iv_reg: sentient::Reg::UNALLOCATED,
                iv: Val(10),
                bound: Val(1),
                carried: vec![carried(Val(3), Val(11), Val(12))],
                dbg_name: None,
                body: vec![
                    inner,
                    Op::Sentient(sentient::Op::ScalarAdd {
                        lhs: Val(22),
                        rhs: Val(4),
                        result: Val(13),
                        reg: Some(Reg {
                            locale: RegType::Lar,
                            index: None,
                        }),
                        element_size: None,
                        ty: ScalarTy::Index,
                    }),
                    Op::Sentient(sentient::Op::Yield {
                        results: vec![Val(13)],
                    }),
                ],
            }),
            load_and_store(Val(12), Val(6), (Val(40), Val(41))),
        ]
    }

    /// `(0 + 2) * 4 = 8` inside, `(8 + 5) * 3 = 39` outside, `+7` for the tail — and the size is the
    /// one memory op unrolled by both bounds, `1 * 4 * 3`.
    #[test]
    fn e018_folds_each_level_then_the_chain_tail_into_the_total_increment() {
        let body = looping_chain(Val(2));
        let regions: [&[Op]; 1] = [&body];
        let defs = Definitions::from_innermost(&regions);
        let mut evaluator = StatedEvaluator {
            held: Vec::new(),
            constants: vec![(Val(4), 5)],
        };
        let found =
            LoopingChainMutableAddrDescriptor::outermost_const_init_and_looping_chain_increment(
                Val(21),
                TransferEnd::Src,
                &body,
                defs,
                &mut evaluator,
            );
        let found = found.expect("the chain resolves to the outer loop's constant init");
        assert_eq!(found.outer_loop, ForRef(Val(10)));
        assert_eq!(found.iter_arg_index, Some(IterArgIndex(0)));
        assert_eq!(found.size, ChainSize(12));
        assert_eq!(evaluator.value(found.increment), 46);
    }

    /// A non-constant bound anywhere in the nest is the reference's all-null tuple (`:3186`). ⭐ THE
    /// DOUBLE IS STILL NEEDED: `getConstant(0)` (`:3161`) precedes the walk, so a refusal is not the
    /// absence of every analysis call.
    #[test]
    fn e018_refuses_a_non_constant_bound() {
        let body = looping_chain(Val(99));
        let regions: [&[Op]; 1] = [&body];
        let defs = Definitions::from_innermost(&regions);
        let mut evaluator = StatedEvaluator::default();
        let found =
            LoopingChainMutableAddrDescriptor::outermost_const_init_and_looping_chain_increment(
                Val(21),
                TransferEnd::Src,
                &body,
                defs,
                &mut evaluator,
            );
        assert_eq!(found, None);
    }

    /// The chain resolves end to end: `is_head_of_chain_` goes true only once the outer loop's init
    /// proved constant, and a twelve-transfer chain that moves by 46 is no single address.
    #[test]
    fn e282_marks_the_head_of_chain_once_the_outer_loops_init_is_constant() {
        let body = looping_chain(Val(2));
        let regions: [&[Op]; 1] = [&body];
        let defs = Definitions::from_innermost(&regions);
        let mut evaluator = StatedEvaluator {
            held: Vec::new(),
            constants: vec![(Val(3), 0), (Val(4), 5)],
        };
        let desc = LoopingChainMutableAddrDescriptor::new(
            Val(21),
            TransferEnd::Src,
            &body,
            defs,
            &mut evaluator,
        );
        assert!(desc.is_valid());
        assert!(desc.is_head_of_chain);
        assert_eq!(desc.outer_loop, Some(ForRef(Val(10))));
        assert_eq!(desc.iter_arg_index, Some(IterArgIndex(0)));
        assert_eq!(desc.size, ChainSize(12));
        assert_eq!(desc.init.map(|init| evaluator.value(init)), Some(0));
        assert_eq!(desc.increment.map(|ev| evaluator.value(ev)), Some(46));
        assert!(!desc.can_be_simplified);
    }
}
