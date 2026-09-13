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
//! | `e017_getOutermostConstInitialization` | 017 | 0 | 107 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2968` |
//! | `e281_DiscreteIntegerSetDescriptor` | 281 | 1 | 45 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2901` |
//! | `e426_dump` | 426 | 2 | 13 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2954` |

use super::write_evaluated_value;
use crate::islands::sentient::dialects::{
    Definitions, Op, Val, operands, regions_ref, results, sentient,
};
use crate::transform::sentient::analyses::{EvaluatedValue, ExpressionEvaluator, MinMax};
use crate::transform::sentient::utils::{ConstKind, is_constant};
use crate::transform::sentient::{ForRef, IterArgIndex};

/// A SET OF ADDRESSES REACHED BY SEVERAL INDEPENDENT INCREMENTS —
/// `class DiscreteIntegerSetDescriptor` (`AddressPinningAndToggle.cpp:454-529`). Unlike
/// [`super::integer_sequence_descriptor::IntegerSequenceDescriptor`] the increments come from a
/// chain of iter args across nested loops, each with its own stride, so the pattern keeps only the
/// initial value and the totals it can move by.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DiscreteIntegerSetDescriptor {
    /// `outer_loop_` — the outermost loop where `%argN` is initialised.
    pub outer_loop: Option<ForRef>,
    /// `iter_arg_index_` — which iter arg of `outer_loop` starts the chain.
    pub iter_arg_index: Option<IterArgIndex>,
    /// `init_` — the initial value.
    pub init: Option<EvaluatedValue>,
    /// `total_positive_delta_` — the sum of `(bound-1) * stride` over the chain's positive strides.
    pub total_positive_delta: Option<EvaluatedValue>,
    /// `total_negative_delta_` — the same sum over its negative strides.
    pub total_negative_delta: Option<EvaluatedValue>,
    /// `can_be_simplified_`, the base class's own field (`AddressPinningAndToggle.cpp:159`) — true
    /// when BOTH delta totals are zero, so the set is really one address.
    pub can_be_simplified: bool,
}

impl DiscreteIntegerSetDescriptor {
    /// Replaces: e426_dump
    ///
    /// The set's initial value and both delta totals as debug text (`:2954-2966`), one `\t`-indented
    /// line each.
    ///
    /// ⛔ TRAP: THE VALID BRANCH ENDS AT `init:` IN [`write_evaluated_value`] — rendering an
    /// `EvaluatedValue` is the out-of-scope analysis's `operator<<`, so only the invalid branch is
    /// complete.
    /// ⛔ AND A VALID DESCRIPTOR CAN CARRY `None` THERE: `isValid()` (`:468`) reads only
    /// `iter_arg_index_` and `outer_loop_`, so the reference's `*init_` here is an unguarded
    /// dereference of a pointer its own validity test never covered.
    /// ⭐ NOTE THE SPACE BEFORE THE FIRST NEWLINE (`"…Descriptor: \n"`): of the six pattern-descriptor dumps
    /// only this one and `LoopingChainMutableAddrDescriptor`'s carry it (`:2956`, `:3123`) — the other
    /// four do not (`:2630`, `:2726`, `:2782`, `:2879`), so it is no family convention to align to.
    #[must_use]
    pub fn dump(&self) -> String {
        let mut out = String::from("Constant Discrete Integer Set Descriptor: \n");
        if !self.is_valid() {
            out.push_str("\tInvalid\n");
            return out;
        }
        if self.can_be_simplified {
            out.push_str("\t(Simplified)\n");
        }
        out.push_str("\tinit:");
        write_evaluated_value(self.init, &mut out);
        out.push_str("\n\ttotal positive delta:");
        write_evaluated_value(self.total_positive_delta, &mut out);
        out.push_str("\n\ttotal negative delta:");
        write_evaluated_value(self.total_negative_delta, &mut out);
        out.push('\n');
        out
    }
}

/// WHAT THE WALK ANSWERS — the reference's
/// `tuple<ForOp, int, const EvaluatedValue *, const EvaluatedValue *>` (`:502-504`).
///
/// ⛔ WHOLE-`None` IS ITS `{nullptr, -1, nullptr, nullptr}`; `iter_arg_index: None` is the OTHER
/// failure it spells with `-1`, where the loop was reached but no constant initialiser was.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutermostConstInitAndDeltas {
    /// The outermost loop the walk reached, named by its induction variable.
    pub outer_loop: ForRef,
    /// Which iter arg of it a constant initialises.
    pub iter_arg_index: Option<IterArgIndex>,
    /// `total_positive_delta` — Σ `max((bound-1) * stride, 0)` over the chain.
    pub total_positive_delta: EvaluatedValue,
    /// `total_negative_delta` — Σ `min((bound-1) * stride, 0)` over the chain.
    pub total_negative_delta: EvaluatedValue,
}

impl DiscreteIntegerSetDescriptor {
    /// Replaces: e281_DiscreteIntegerSetDescriptor
    ///
    /// Matches `base_addr` as the iter arg of a loop nest whose outermost initialisation is constant,
    /// recording that loop, its index, the initial value and both delta totals (`:2901-2951`).
    ///
    /// ⛔ IT `invalidate()`s ON THE INDEX FAILURE (`:2933`) where its `IntegerSequenceDescriptor`
    /// twin (e280) returns untouched — the two constructors differ exactly there.
    /// ⭐ THE DELTAS COME FROM ITS OWN MEMBER WALK (e017), never from `dcc::utils`'.
    #[must_use]
    pub fn new(
        base_addr: Val,
        defs: Definitions<'_>,
        evaluator: &mut impl ExpressionEvaluator,
    ) -> DiscreteIntegerSetDescriptor {
        // `dyn_cast<BlockArgument>(base_addr)` (`:2909`) and `dyn_cast<ForOp>(getParentOp())`
        // (`:2916`) are one lookup — `sentient.for` is the only op of this island that binds a
        // region argument at all.
        if defs.for_arg_of(base_addr).is_none() {
            return DiscreteIntegerSetDescriptor::default();
        }

        let found = DiscreteIntegerSetDescriptor::outermost_const_initialization(
            base_addr, defs, evaluator,
        );
        let mut desc = match found {
            None => DiscreteIntegerSetDescriptor::default(),
            Some(found) => DiscreteIntegerSetDescriptor {
                outer_loop: Some(found.outer_loop),
                iter_arg_index: found.iter_arg_index,
                init: None,
                total_positive_delta: Some(found.total_positive_delta),
                total_negative_delta: Some(found.total_negative_delta),
                can_be_simplified: false,
            },
        };

        // `if (!outer_loop_ || iter_arg_index_ < 0) { invalidate(); return; }` (`:2929-2936`).
        let (Some(outer_loop), Some(iter_arg_index)) = (desc.outer_loop, desc.iter_arg_index)
        else {
            desc.invalidate();
            return desc;
        };
        // `Value curr_init = outer_loop_.getIterOperands()[iter_arg_index_]` (`:2938`).
        let Some(curr_init) = super::iter_operand_of(outer_loop, iter_arg_index, defs) else {
            desc.invalidate();
            return desc;
        };
        if !is_constant(curr_init, ConstKind::ScalarConstant, defs) {
            desc.invalidate();
            return desc;
        }
        desc.init = Some(evaluator.evaluate_value_handle(curr_init));

        let zero = evaluator.constant(0);
        desc.can_be_simplified =
            desc.total_positive_delta == Some(zero) && desc.total_negative_delta == Some(zero);
        desc
    }

    /// Replaces: e017_getOutermostConstInitialization
    ///
    /// Walks the iter-arg initialisation chain inner to outer, accumulating each loop's
    /// `(bound-1) * stride` into a positive and a negative total (`:2968-3076`).
    ///
    /// ⛔ NOT THE SAME FUNCTION AS [`crate::transform::sentient::utils::outermost_const_initialization`]
    /// (`Utils.cpp:469`), which answers a SIZE — this one answers per-unit deltas and refuses a
    /// non-constant bound outright instead of recording it.
    /// ⛔ EVERY REFUSAL IS `None`, INCLUDING `bound <= 0` (`:3002`): the reference's own comment
    /// there is "dead load/stores?".
    #[must_use]
    pub fn outermost_const_initialization(
        iter_arg: Val,
        defs: Definitions<'_>,
        evaluator: &mut impl ExpressionEvaluator,
    ) -> Option<OutermostConstInitAndDeltas> {
        let mut iter_arg_index: Option<IterArgIndex> = None;
        let mut curr: Option<Val> = Some(iter_arg);
        let mut outer_loop: Option<&Op> = None;
        let mut totals: Option<(EvaluatedValue, EvaluatedValue)> = None;

        // `while (iter_arg_index < 0 && cur_iter_arg)` (`:2993`)
        while iter_arg_index.is_none() {
            let Some(curr_arg) = curr else { break };
            // `getIndexOfLoopRegionIterArgs` then `cur_iter_arg.getOwner()->getParentOp()`, whose
            // `dyn_cast<ForOp>` the reference then uses UNCHECKED (`:2994-2996`).
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

            // `dyn_cast_or_null<ConstantOp>(outer_loop.getBound().getDefiningOp())` — ⛔ THE OP
            // ITSELF, not `isConstant`, so a `uniform.query_map` bound is refused here.
            let Some(Op::Sentient(sentient::Op::ScalarConstant {
                value: bound_value, ..
            })) = defs.of(*bound)
            else {
                return None;
            };
            if *bound_value <= 0 {
                return None;
            }

            // `getIterArgIncrementer<ForOp, AddOp>` and the constancy of its `getOperand(1)`.
            let (add_result, stride_val) = iter_arg_incrementer(carried, body, curr_it_index)?;
            if !is_constant(stride_val, ConstKind::ScalarConstant, defs) {
                return None;
            }
            let stride = evaluator.evaluate_value_handle(stride_val);

            // `cur_loop_delta = stride * (bound-1)`, split at zero into its positive and negative
            // halves so that the two run in opposite directions (`:3035-3046`).
            let cur_loop_delta = evaluator.evaluate_multiply_by_const(stride, bound_value - 1);
            let zero = evaluator.constant(0);
            let cur_positive = evaluator.evaluate_min_max(&[cur_loop_delta, zero], MinMax::Max);
            let cur_negative = evaluator.evaluate_min_max(&[cur_loop_delta, zero], MinMax::Min);
            totals = Some(match totals {
                None => (cur_positive, cur_negative),
                Some((positive, negative)) => (
                    evaluator.evaluate_sum_handle(positive, cur_positive),
                    evaluator.evaluate_sum_handle(negative, cur_negative),
                ),
            });

            // `DT_CHECK_MSG(num_non_yield_feeding_uses == 1, ..)` (`:3064-3067`) — an ABORT in the
            // reference, so it stays a named stop rather than becoming a refusal.
            let uses = num_users_except(curr_arg, body, &|op| {
                matches!(op, Op::Sentient(sentient::Op::Yield { .. }))
                    || results(op).contains(&add_result)
            });
            if uses != 1 {
                todo!(
                    "getOutermostConstInitialization: DT_CHECK_MSG(num_non_yield_feeding_uses == \
                     1, \"base_addr should only be used in memory op and to feed the yield op\") \
                     — {uses} such uses of {curr_arg:?} (:3064-3067)"
                )
            }

            let curr_init = carried.get(curr_it_index)?.init;
            if defs.for_arg_of(curr_init).is_some() {
                curr = Some(curr_init);
            } else {
                if is_constant(curr_init, ConstKind::ScalarConstant, defs) {
                    iter_arg_index = u32::try_from(curr_it_index).ok().map(IterArgIndex);
                }
                curr = None;
            }
        }

        let Op::Sentient(sentient::Op::For { iv, .. }) = outer_loop? else {
            return None;
        };
        let (total_positive_delta, total_negative_delta) = totals?;
        Some(OutermostConstInitAndDeltas {
            outer_loop: ForRef(*iv),
            iter_arg_index,
            total_positive_delta,
            total_negative_delta,
        })
    }
}

/// `dcc::utils::getIterArgIncrementer<sentient::ForOp, sentient::AddOp>`
/// (`dcc/src/Utils/Utils.cpp:342-355`) — the `sentient.scalar_add` that advances iter arg `index`,
/// as `(its result, its second operand)`. `None` covers both the reference's `nullptr` and the
/// `DT_CHECK_MSG(idx < ..getNumOperands(), "index is outside operand range")` a loop whose yield is
/// short of its carried list would trip.
pub(super) fn iter_arg_incrementer(
    carried: &[sentient::Carried],
    body: &[Op],
    index: usize,
) -> Option<(Val, Val)> {
    let arg = carried.get(index)?.arg;
    let Op::Sentient(sentient::Op::Yield { results: yielded }) = body.last()? else {
        return None;
    };
    let yield_operand = *yielded.get(index)?;
    let add = body
        .iter()
        .find(|op| results(op).contains(&yield_operand))?;
    let Op::Sentient(sentient::Op::ScalarAdd {
        lhs, rhs, result, ..
    }) = add
    else {
        return None;
    };
    (*lhs == arg).then_some((*result, *rhs))
}

/// `dcc::utils::getNumUsersExcept` (`dcc/src/Utils/Utils.cpp:384-390`) — uses of `val` in `scope`
/// and its nested regions whose owner `filter_out` rejects, counted ONCE PER USE as `getUsers()`
/// iterates uses rather than distinct ops.
pub(super) fn num_users_except(val: Val, scope: &[Op], filter_out: &impl Fn(&Op) -> bool) -> usize {
    scope
        .iter()
        .map(|op| {
            let here = if filter_out(op) {
                0
            } else {
                operands(op).iter().filter(|read| **read == val).count()
            };
            here + regions_ref(op)
                .into_iter()
                .map(|region| num_users_except(val, region, filter_out))
                .sum::<usize>()
        })
        .sum()
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
    use crate::transform::sentient::analyses::{
        Evaluation, OffsetSites, OutOfScopeEvaluator, ScalarOffset,
    };

    /// The handle flavour with its answers stated as INTEGERS: one entry per interned value, so `==`
    /// on handles is `==` on values and the walk's arithmetic is observable.
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
            todo!("e017 asks for handles, never for a decoded evaluation")
        }

        fn evaluate_sum(&mut self, _lhs: &Evaluation, _rhs: &Evaluation) -> Evaluation {
            todo!("e017 asks for handles, never for a decoded evaluation")
        }

        fn build_offset_value(
            &mut self,
            _evaluation: &Evaluation,
            _sites: &mut OffsetSites<'_>,
            _walked: &mut Vec<Op>,
            _ty: ScalarTy,
        ) -> Val {
            todo!("e017 builds no value")
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

        fn evaluate_multiply_by_const(&mut self, ev: EvaluatedValue, by: i64) -> EvaluatedValue {
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

    fn add(lhs: Val, rhs: Val, result: Val) -> Op {
        Op::Sentient(sentient::Op::ScalarAdd {
            lhs,
            rhs,
            result,
            reg: Some(Reg {
                locale: RegType::Lar,
                index: None,
            }),
            element_size: None,
            ty: ScalarTy::Index,
        })
    }

    fn carried(init: Val, arg: Val, result: Val) -> Carried {
        Carried {
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

    fn load_and_store(src_mutable_addr: Val) -> Op {
        Op::Sentient(sentient::Op::LoadAndStore {
            src: Val(90),
            dst: Val(91),
            src_mutable_addr,
            src_immutable_addr: Val(92),
            src_inc: Val(93),
            dst_mutable_addr: Val(94),
            dst_immutable_addr: Val(95),
            dst_inc: Val(93),
            multicast_info: None,
            results: (Val(96), Val(97)),
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

    /// The reference's own example (`:437-452`): `%arg1` starts at a constant and two nested loops add
    /// `+3` and `-4` to the chain, `outer_bound` being the OUTER loop's.
    fn chained_nest(outer_bound: Val) -> Vec<Op> {
        let inner = Op::Sentient(sentient::Op::For {
            iv: Val(20),
            bound: Val(2),
            bound_reg: None,
            carried: vec![carried(Val(11), Val(21), Val(22))],
            dbg_name: None,
            body: vec![
                load_and_store(Val(21)),
                add(Val(21), Val(4), Val(23)),
                Op::Sentient(sentient::Op::Yield {
                    results: vec![Val(23)],
                }),
            ],
        });
        vec![
            constant(3, Val(1)),
            constant(5, Val(2)),
            constant(0, Val(3)),
            constant(3, Val(4)),
            constant(-4, Val(5)),
            Op::Sentient(sentient::Op::For {
                iv: Val(10),
                bound: outer_bound,
                bound_reg: None,
                carried: vec![carried(Val(3), Val(11), Val(12))],
                dbg_name: None,
                body: vec![
                    inner,
                    add(Val(11), Val(5), Val(13)),
                    Op::Sentient(sentient::Op::Yield {
                        results: vec![Val(13)],
                    }),
                ],
            }),
        ]
    }

    /// The deltas are `Σ max((bound-1) * stride, 0)` and `Σ min(…, 0)`: `+3` over five iterations and
    /// `-4` over three give `+12` and `-8`, and the walk ends at the OUTER loop's constant init.
    #[test]
    fn e017_accumulates_the_positive_and_negative_deltas_of_the_chain() {
        let body = chained_nest(Val(1));
        let regions: [&[Op]; 1] = [&body];
        let defs = Definitions::from_innermost(&regions);
        let mut evaluator = StatedEvaluator {
            held: Vec::new(),
            constants: vec![(Val(4), 3), (Val(5), -4)],
        };
        let found = DiscreteIntegerSetDescriptor::outermost_const_initialization(
            Val(21),
            defs,
            &mut evaluator,
        );
        let found = found.expect("the chain resolves to the outer loop's constant init");
        assert_eq!(found.outer_loop, ForRef(Val(10)));
        assert_eq!(found.iter_arg_index, Some(IterArgIndex(0)));
        assert_eq!(evaluator.value(found.total_positive_delta), 12);
        assert_eq!(evaluator.value(found.total_negative_delta), -8);
    }

    /// `dyn_cast_or_null<ConstantOp>(outer_loop.getBound()...)` failing is a REFUSAL, not an abort:
    /// the walk answers nothing without asking the analysis anything, which
    /// [`OutOfScopeEvaluator`] proves by panicking if it is asked.
    #[test]
    fn e017_refuses_a_non_constant_bound_without_asking_the_analysis() {
        let body = chained_nest(Val(99));
        let regions: [&[Op]; 1] = [&body];
        let defs = Definitions::from_innermost(&regions);
        let found = DiscreteIntegerSetDescriptor::outermost_const_initialization(
            Val(11),
            defs,
            &mut OutOfScopeEvaluator::default(),
        );
        assert_eq!(found, None);
    }

    /// The reference's own example (`:437-452`) end to end: the walk's deltas are kept, the outer
    /// loop's constant init becomes `init_`, and a set that moves both ways is no single address.
    #[test]
    fn e281_keeps_the_walks_deltas_and_the_outer_loops_constant_init() {
        let body = chained_nest(Val(1));
        let regions: [&[Op]; 1] = [&body];
        let defs = Definitions::from_innermost(&regions);
        let mut evaluator = StatedEvaluator {
            held: Vec::new(),
            constants: vec![(Val(3), 0), (Val(4), 3), (Val(5), -4)],
        };
        let desc = DiscreteIntegerSetDescriptor::new(Val(21), defs, &mut evaluator);
        assert!(desc.is_valid());
        assert_eq!(desc.outer_loop, Some(ForRef(Val(10))));
        assert_eq!(desc.iter_arg_index, Some(IterArgIndex(0)));
        assert_eq!(desc.init.map(|init| evaluator.value(init)), Some(0));
        assert_eq!(
            desc.total_positive_delta.map(|ev| evaluator.value(ev)),
            Some(12)
        );
        assert_eq!(
            desc.total_negative_delta.map(|ev| evaluator.value(ev)),
            Some(-8)
        );
        assert!(!desc.can_be_simplified);
    }

    /// 426/656 — the complete branch: no recognised pattern, and the whole trace is the header (note
    /// its trailing space) and the word.
    #[test]
    fn e426_an_invalid_descriptor_dumps_the_header_and_invalid() {
        let desc = DiscreteIntegerSetDescriptor::default();
        assert_eq!(
            desc.dump(),
            "Constant Discrete Integer Set Descriptor: \n\tInvalid\n"
        );
    }

    /// The vendor's own case: a matched set reaches `init:` and stops in the analysis's own
    /// `operator<<`. ⛔ Nothing written before the stop is observable — the `String` dies with it.
    #[test]
    #[should_panic(expected = "EvaluatedValue::operator<<")]
    fn e426_a_matched_descriptor_stops_at_the_out_of_scope_rendering() {
        let desc = DiscreteIntegerSetDescriptor {
            outer_loop: Some(ForRef(Val(10))),
            iter_arg_index: Some(IterArgIndex(0)),
            init: Some(EvaluatedValue(1)),
            total_positive_delta: Some(EvaluatedValue(2)),
            total_negative_delta: Some(EvaluatedValue(3)),
            can_be_simplified: false,
        };
        let _ = desc.dump();
    }
}
