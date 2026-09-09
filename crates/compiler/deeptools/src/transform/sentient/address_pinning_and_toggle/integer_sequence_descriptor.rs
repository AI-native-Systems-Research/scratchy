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

//! `AddressPinningAndToggle.cpp` — 2 of the campaign's 656 units (dependency level(s) [1, 3]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e280_IntegerSequenceDescriptor` | 280 | 1 | 75 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2799` |
//! | `e493_dump` | 493 | 3 | 20 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2877` |


// crustify:todo: e493_dump
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2877  (20 body lines, level 3)
//   original  : void IntegerSequenceDescriptor::dump() const
//   calls     : e002_getAllConstants, e252_size, e278_isValid, e279_canBeSimplified, e408_getAllConstants


use crate::islands::sentient::dialects::{Definitions, Op, Val, sentient};
use crate::transform::sentient::analyses::{EvaluatedValue, ExpressionEvaluator};
use crate::transform::sentient::utils::{ChainSize, ConstKind, is_constant};
use crate::transform::sentient::{ForRef, IterArgIndex};

/// HOW MANY TERMS AN INTEGER SEQUENCE HAS — `int size_`, which is TRI-STATE in the reference.
///
/// ⛔ `-1` IS NOT A COUNT AND `0` IS NOT ONE EITHER. `size_` is documented as "can be -1 if its part
/// of a symbolic loop" (`AddressPinningAndToggle.cpp:461`) and `isValid()` requires `size_ > 0`
/// (`:392`), so both non-positive values mean "no sequence" for different reasons — an enum states
/// that where a signed count would leave `-1` to arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SequenceSize {
    /// `size_ == 0` — never matched, or cleared by [`IntegerSequenceDescriptor::invalidate`].
    #[default]
    Cleared,
    /// `size_ == -1` — the sequence sits in a symbolic loop, so its length is not a compile-time
    /// number. ⛔ Still invalid: `isValid()` reads `size_ > 0`.
    Symbolic,
    /// `size_ > 0` — the number of terms.
    Terms(u32),
}

/// A CONSTANT INTEGER SEQUENCE FORMED BY A STATIC LOOP'S ITER ARG —
/// `class IntegerSequenceDescriptor` (`AddressPinningAndToggle.cpp:345-464`): `init_ + stride_ * i`
/// for `i` in `0..size_`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IntegerSequenceDescriptor {
    /// `outer_loop_` — the outermost loop where `%argN` is initialised.
    pub outer_loop: Option<ForRef>,
    /// `iter_arg_index_` — which iter arg of `outer_loop` walks the sequence.
    pub iter_arg_index: Option<IterArgIndex>,
    /// `init_` — the first term.
    pub init: Option<EvaluatedValue>,
    /// `stride_` — the step between terms.
    pub stride: Option<EvaluatedValue>,
    /// `size_` — how many terms. See [`SequenceSize`].
    pub size: SequenceSize,
    /// `can_be_simplified_`, the base class's own field (`AddressPinningAndToggle.cpp:159`) — true
    /// when the sequence has one term or never moves, so it is really one constant.
    pub can_be_simplified: bool,
}

impl IntegerSequenceDescriptor {
    /// Replaces: e280_IntegerSequenceDescriptor
    ///
    /// Matches `base_addr` as a loop iter arg walking `init + stride * i`, recording the outermost
    /// loop, its index, the term count, the first term and the stride (`:2799-2874`).
    ///
    /// ⛔ THE FIRST TWO EXITS DO NOT `invalidate()` (`:2825`, `:2833`) — the loop, index and size the
    /// walk already assigned SURVIVE them, and only a bad init or stride clears everything.
    /// ⛔ [`SequenceSize::Symbolic`] IS THE ONLY ROUTE TO `size_ <= 0`: a `bound <= 0` makes
    /// `outermost_const_initialization` answer nothing at all, which is the `!outer_loop_` exit.
    /// ⭐ THE STRIDE COMES FROM `base_addr`'s OWN LOOP, not from `outer_loop_` (`:2857-2861`).
    #[must_use]
    pub fn new(
        base_addr: Val,
        defs: Definitions<'_>,
        evaluator: &mut impl ExpressionEvaluator,
    ) -> IntegerSequenceDescriptor {
        // `dyn_cast<BlockArgument>(base_addr)` (`:2807`) and `iter_arg.getOwner()->getParentOp()`
        // (`:2814`) are one lookup: a region argument of this island has a `sentient.for` parent or
        // it is not a region argument at all. `getIndexOfLoopRegionIterArgs` is the `- 1`.
        let Some((loop_op, arg_number)) = defs.for_arg_of(base_addr) else {
            return IntegerSequenceDescriptor::default();
        };
        let Some(inner_loop_iter_arg_index) = arg_number.checked_sub(1) else {
            return IntegerSequenceDescriptor::default();
        };

        // `std::tie(outer_loop_, iter_arg_index_, size_) = getOutermostConstInitialization(iter_arg)`
        let found =
            crate::transform::sentient::utils::outermost_const_initialization(base_addr, defs);
        let mut desc = match found {
            None => IntegerSequenceDescriptor::default(),
            Some(found) => IntegerSequenceDescriptor {
                outer_loop: Some(found.loop_op),
                iter_arg_index: found.iter_arg,
                init: None,
                stride: None,
                size: match found.size {
                    ChainSize::Unknown => SequenceSize::Symbolic,
                    ChainSize::Iterations(iterations) => {
                        SequenceSize::Terms(u32::try_from(iterations.get()).unwrap_or(u32::MAX))
                    }
                },
                can_be_simplified: false,
            },
        };

        // `if (!outer_loop_ || iter_arg_index_ < 0) return;` (`:2825`) and `if (size_ <= 0) return;`
        // (`:2833`) — both keep what is already assigned.
        let (Some(outer_loop), Some(iter_arg_index)) = (desc.outer_loop, desc.iter_arg_index)
        else {
            return desc;
        };
        if !matches!(desc.size, SequenceSize::Terms(terms) if terms > 0) {
            return desc;
        }

        // `Value curr_init = outer_loop_.getIterOperands()[iter_arg_index_]` (`:2841`) — the
        // reference indexes unchecked, and an index that walk produced is in range by construction.
        let Some(curr_init) = super::iter_operand_of(outer_loop, iter_arg_index, defs) else {
            desc.invalidate();
            return desc;
        };
        if !is_constant(curr_init, ConstKind::ScalarConstant, defs) {
            desc.invalidate();
            return desc;
        }
        desc.init = Some(evaluator.evaluate_value_handle(curr_init));

        // `getIterArgIncrementer<ForOp, AddOp>(loop_op, inner_loop_iter_arg_index)` and the
        // constancy of its `getOperand(1)` (`:2857-2868`).
        let Op::Sentient(sentient::Op::For { carried, body, .. }) = loop_op else {
            desc.invalidate();
            return desc;
        };
        let incrementer = super::discrete_integer_set_descriptor::iter_arg_incrementer(
            carried,
            body,
            inner_loop_iter_arg_index,
        );
        let Some((add_result, stride_val)) = incrementer else {
            desc.invalidate();
            return desc;
        };
        if !is_constant(stride_val, ConstKind::ScalarConstant, defs) {
            desc.invalidate();
            return desc;
        }
        let stride = evaluator.evaluate_value_handle(stride_val);
        desc.stride = Some(stride);

        // `DT_CHECK_MSG(num_non_yield_feeding_uses == 1, ..)` (`:2870-2873`) — an ABORT in the
        // reference, so it stays a named stop rather than becoming a refusal.
        let uses =
            super::discrete_integer_set_descriptor::num_users_except(base_addr, body, &|op| {
                matches!(op, Op::Sentient(sentient::Op::Yield { .. }))
                    || crate::islands::sentient::dialects::results(op).contains(&add_result)
            });
        if uses != 1 {
            todo!(
                "IntegerSequenceDescriptor: DT_CHECK_MSG(num_non_yield_feeding_uses == 1, \
                 \"base_addr should only be used in memory op and to feed the yield op\") — \
                 {uses} such uses of {base_addr:?} (:2870-2873)"
            )
        }

        let zero = evaluator.constant(0);
        desc.can_be_simplified = desc.size == SequenceSize::Terms(1) || stride == zero;
        desc
    }
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
    use crate::transform::sentient::analyses::{Evaluation, OffsetSites, OutOfScopeEvaluator};

    /// The handle flavour with its answers stated as INTEGERS, so `==` on handles is `==` on values.
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
            todo!("e280 asks for handles, never for a decoded evaluation")
        }

        fn evaluate_sum(&mut self, _lhs: &Evaluation, _rhs: &Evaluation) -> Evaluation {
            todo!("e280 never sums")
        }

        fn build_offset_value(
            &mut self,
            _evaluation: &Evaluation,
            _sites: &mut OffsetSites<'_>,
            _walked: &mut Vec<Op>,
            _ty: ScalarTy,
        ) -> Val {
            todo!("e280 builds no value")
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

    /// The reference's own example (`:345-352`): `%arg1` starts at a constant `0` and one loop of
    /// `bound` iterations adds `+3` each time, the transfer being `%arg1`'s only other use.
    fn sequence(bound: Val) -> Vec<Op> {
        vec![
            constant(0, Val(1)),
            constant(5, Val(2)),
            constant(3, Val(3)),
            Op::Sentient(sentient::Op::For {
                iv: Val(10),
                bound,
                carried: vec![Carried {
                    init: Val(1),
                    arg: Val(11),
                    result: Val(12),
                    reg: Reg {
                        locale: RegType::Lar,
                        index: None,
                    },
                    program_header: false,
                    element_size: None,
                }],
                dbg_name: None,
                body: vec![
                    load_and_store(Val(11)),
                    add(Val(11), Val(3), Val(13)),
                    Op::Sentient(sentient::Op::Yield {
                        results: vec![Val(13)],
                    }),
                ],
            }),
        ]
    }

    /// Five terms from `0` by `+3`: neither one term nor a zero stride, so it is a real sequence.
    #[test]
    fn e280_records_the_loop_the_size_the_first_term_and_the_stride() {
        let body = sequence(Val(2));
        let regions: [&[Op]; 1] = [&body];
        let defs = Definitions::from_innermost(&regions);
        let mut evaluator = StatedEvaluator {
            held: Vec::new(),
            constants: vec![(Val(1), 0), (Val(3), 3)],
        };
        let desc = IntegerSequenceDescriptor::new(Val(11), defs, &mut evaluator);
        assert!(desc.is_valid());
        assert_eq!(desc.outer_loop, Some(ForRef(Val(10))));
        assert_eq!(desc.iter_arg_index, Some(IterArgIndex(0)));
        assert_eq!(desc.size, SequenceSize::Terms(5));
        assert_eq!(desc.init.map(|init| evaluator.value(init)), Some(0));
        assert_eq!(desc.stride.map(|stride| evaluator.value(stride)), Some(3));
        assert!(!desc.can_be_simplified);
    }

    /// A non-constant bound makes the size unknown, and the reference RETURNS THERE WITHOUT
    /// `invalidate()` — the loop and index it already assigned survive, `init` and `stride` do not,
    /// and [`OutOfScopeEvaluator`] proves the analysis was never asked.
    #[test]
    fn e280_keeps_the_loop_and_index_when_the_size_is_symbolic() {
        let body = sequence(Val(99));
        let regions: [&[Op]; 1] = [&body];
        let defs = Definitions::from_innermost(&regions);
        let desc =
            IntegerSequenceDescriptor::new(Val(11), defs, &mut OutOfScopeEvaluator::default());
        assert!(!desc.is_valid());
        assert_eq!(desc.outer_loop, Some(ForRef(Val(10))));
        assert_eq!(desc.iter_arg_index, Some(IterArgIndex(0)));
        assert_eq!(desc.size, SequenceSize::Symbolic);
        assert_eq!(desc.init, None);
        assert_eq!(desc.stride, None);
    }
}
