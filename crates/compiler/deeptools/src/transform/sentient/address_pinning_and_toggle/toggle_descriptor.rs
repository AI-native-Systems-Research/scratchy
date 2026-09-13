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

//! `AddressPinningAndToggle.cpp` — 3 of the campaign's 656 units (dependency level(s) [0, 4]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e015_getInit` | 015 | 0 | 10 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2713` |
//! | `e552_ToggleDescriptor` | 552 | 4 | 68 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2642` |
//! | `e553_dump` | 553 | 4 | 16 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2724` |

use super::discrete_integer_set_descriptor::num_users_except;
use super::write_evaluated_value;
use crate::islands::sentient::dialects::{Definitions, Op, Val, sentient};
use crate::transform::sentient::analyses::{EvaluatedValue, ExpressionEvaluator};
use crate::transform::sentient::utils::{ConstKind, is_constant, outermost_const_initialization};
use crate::transform::sentient::{ForRef, IterArgIndex};

/// A BASE ADDRESS TOGGLING BETWEEN TWO CONSTANTS — `class ToggleDescriptor`
/// (`AddressPinningAndToggle.cpp:213-286`), matched as `X = c1 - Y` around a loop's iter arg.
///
/// ⛔ ITS FIRST THREE FIELDS ARE EXACTLY WHAT `isValid()` READS (`:263`:
/// `c1_ && iter_arg_index_ >= 0 && outer_loop_`), which is why [`ToggleDescriptor::invalidate`]
/// clears these three and nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ToggleDescriptor {
    /// `outer_loop_` — the outermost loop where `%argN` is initialised.
    pub outer_loop: Option<ForRef>,
    /// `iter_arg_index_` — which iter arg of `outer_loop` carries the toggle.
    pub iter_arg_index: Option<IterArgIndex>,
    /// `c1_` — the constant term in `X = c1 - Y`.
    pub c1: Option<EvaluatedValue>,
    /// `can_be_simplified_` — the base class's cached flag (`:159`), which the constructor sets to
    /// `getX() == getY()` (e552, `:2710`): a "toggle" whose two constants turned out to be one.
    ///
    /// ⛔ IT IS PART OF THE TRANSFER'S VALIDITY, not a hint. `DataTransferDescriptor::isValid()`
    /// accepts a toggle holding a SINGLE base address only when this is true (`:2484`), which is why
    /// [`super::DataTransferDescriptor::is_toggle`] reads it.
    pub can_be_simplified: bool,
}

impl ToggleDescriptor {
    /// Replaces: e552_ToggleDescriptor
    ///
    /// Matches `base_addr` as `c1 - %argN` around a loop whose outermost initialisation is constant,
    /// recording that loop, its index and `c1`, then flagging a "toggle" whose two values are one
    /// (`:2642-2711`).
    ///
    /// ⛔ THE LADDER HAS TWO DIFFERENT FAILURES. The four shape tests return the descriptor UNTOUCHED
    /// (`:2646`, `:2653`, `:2662`, `:2669` — default, and already invalid); the two later ones
    /// `invalidate()` fields the walk had filled (`:2680`, `:2691`).
    /// ⛔ `setCanBeSimplified(getX() == getY())` COMPARES VALUES, NOT HANDLES (`:2710`) —
    /// `EvaluatedValue::operator==`, which is [`ExpressionEvaluator::values_equal`].
    #[must_use]
    pub fn new(
        base_addr: Val,
        defs: Definitions<'_>,
        evaluator: &mut impl ExpressionEvaluator,
    ) -> ToggleDescriptor {
        // `isa<BlockArgument>(base_addr)` (`:2646`) — `sentient.for` is the only op of this island that
        // binds one, so the lookup that answers it is the same one `:2669` needs.
        if defs.for_arg_of(base_addr).is_some() {
            return ToggleDescriptor::default();
        }
        // `base_addr.getDefiningOp<sentient::SubOp>()` (`:2653`).
        let Some(Op::Sentient(sentient::Op::ScalarSub {
            lhs: inp1,
            rhs: inp2,
            ..
        })) = defs.of(base_addr)
        else {
            return ToggleDescriptor::default();
        };
        // `dyn_cast<BlockArgument>(sub.getInp2())` AND `isConstant<ConstantOp>(sub.getInp1())`
        // (`:2657-2662`), then `dyn_cast<ForOp>(iter_arg.getOwner()->getParentOp())` (`:2666-2669`).
        let iter_arg = *inp2;
        let Some((for_op, arg_number)) = defs.for_arg_of(iter_arg) else {
            return ToggleDescriptor::default();
        };
        if !is_constant(*inp1, ConstKind::ScalarConstant, defs) {
            return ToggleDescriptor::default();
        }
        let c1_operand = *inp1;

        // `std::tie(outer_loop_, iter_arg_index_, std::ignore) = getOutermostConstInitialization(..)`
        // (`:2671-2672`); whole-`None` is its `{nullptr, -1, size}`.
        let found = outermost_const_initialization(iter_arg, defs);
        let mut desc = ToggleDescriptor {
            outer_loop: found.map(|found| found.loop_op),
            iter_arg_index: found.and_then(|found| found.iter_arg),
            c1: None,
            can_be_simplified: false,
        };
        // `if (iter_arg_index_ < 0) { invalidate(); return; }` (`:2674-2681`).
        if desc.iter_arg_index.is_none() {
            desc.invalidate();
            return desc;
        }

        // `int iter_arg_index = getIndexOfLoopRegionIterArgs(iter_arg)` (`:2683`) — a LOCAL and NOT the
        // member above: `getRegionIterArgs()` drops the induction variable, so it is one less than the
        // raw argument number (`Analyses/Utils.cpp:257-271`).
        let Some(region_iter_arg_index) = arg_number.checked_sub(1) else {
            desc.invalidate();
            return desc;
        };
        let Op::Sentient(sentient::Op::For { body, .. }) = for_op else {
            desc.invalidate();
            return desc;
        };
        // `DT_CHECK_MSG(yield, "expected terminator of loop body to be a yield")` (`:2684-2685`).
        let Some(Op::Sentient(sentient::Op::Yield { results })) = body.last() else {
            todo!(
                "ToggleDescriptor: DT_CHECK_MSG(yield, \"expected terminator of loop body to be a \
                 yield\") (:2684-2685) — the loop binding {iter_arg:?} ends in {:?}",
                body.last()
            )
        };
        // `if (yield.getOperand(iter_arg_index) != base_addr) { invalidate(); return; }` (`:2686-2693`).
        if results.get(region_iter_arg_index) != Some(&base_addr) {
            desc.invalidate();
            return desc;
        }

        desc.c1 = Some(evaluator.evaluate_value_handle(c1_operand));
        // `DT_CHECK_MSG(num_non_yield_feeding_uses == 1, ..)` (`:2696-2700`) — an ABORT there, so a
        // named stop here. ⚠️ DIVERGENCE: the reference filters the ONE terminator by identity and this
        // filters every `sentient.yield`, as its e017 sibling already does.
        let uses = num_users_except(base_addr, body, &|op| {
            matches!(op, Op::Sentient(sentient::Op::Yield { .. }))
        });
        if uses != 1 {
            todo!(
                "ToggleDescriptor: DT_CHECK_MSG(num_non_yield_feeding_uses == 1, \"base_addr should \
                 only be used in memory op and to feed the yield op\") — {uses} such uses of \
                 {base_addr:?} (:2696-2700)"
            )
        }

        // `setCanBeSimplified(getX() == getY())` (`:2710`).
        let x = desc.x(evaluator, body, defs);
        let y = desc.init(body, defs);
        desc.can_be_simplified = match x {
            Some(x) => evaluator.values_equal(x, y),
            None => false,
        };
        desc
    }

    /// Replaces: e553_dump
    ///
    /// This toggle as debug text (`:2724-2739`) — `\t`-indented, `X`/`Y`/`c1`, the outer loop's nest
    /// level and its iter-arg index.
    ///
    /// ⛔ TRAP: THE VALID BRANCH ENDS IN [`write_evaluated_value`], so only the `Invalid` branch is
    /// complete — the same seam as e425's.
    /// ⛔ `(Simplified)` PRECEDES THE VALUES AND DOES NOT REPLACE THEM (`:2731`).
    #[must_use]
    pub fn dump(
        &self,
        body: &[Op],
        defs: Definitions<'_>,
        evaluator: &mut impl ExpressionEvaluator,
    ) -> String {
        let mut out = String::from("Toggle Descriptor:\n");
        // `const char indent = '\t'` (`:2727`), streamed before each line the reference indents.
        if !self.is_valid() {
            out.push_str("\tInvalid\n");
            return out;
        }
        if self.can_be_simplified {
            out.push_str("\t(Simplified)\n");
        }
        out.push_str("\tX = ");
        write_evaluated_value(self.x(evaluator, body, defs), &mut out);
        out.push_str("\n\tY = ");
        write_evaluated_value(Some(self.init(body, defs)), &mut out);
        out.push_str("\n\tc1 = ");
        write_evaluated_value(self.c1, &mut out);
        out.push_str("\n\touter loop at level (");
        // `-1` is the reference's initial `level`, unreachable past the `isValid()` above.
        out.push_str(&self.outer_loop.map_or(-1, |l| loop_nest_level(l, body)).to_string());
        out.push_str(")\n\titer arg index of outer loop: ");
        out.push_str(&match self.iter_arg_index {
            Some(index) => index.0.to_string(),
            None => "-1".to_owned(),
        });
        out.push('\n');
        out
    }

    /// Replaces: e015_getInit
    ///
    /// The evaluated constant the outer loop's iter arg starts at —
    /// `getIterOperands()[iter_arg_index_]`.
    /// ⭐ BOTH `DT_CHECK`s AT `:2714-2715` ARE THE TWO `Option`s THIS TYPE ALREADY CARRIES:
    /// `iter_arg_index_ >= 0` and `outer_loop_ != nullptr` are absence, not comparisons.
    /// ⛔ ONLY THE EVALUATION IS MISSING: `evaluator_.evaluateValue(init)` belongs to
    /// `ExpressionEvaluator` (`Analyses/ExpressionEvaluatorUtils`, out of campaign scope), so it is a
    /// `todo!` — resolving the operand and proving it constant, everything before that call, is here.
    #[must_use]
    pub fn init(&self, body: &[Op], defs: Definitions<'_>) -> EvaluatedValue {
        let (Some(outer_loop), Some(iter_arg_index)) = (self.outer_loop, self.iter_arg_index)
        else {
            todo!(
                "getInit on an invalid ToggleDescriptor — `DT_CHECK(iter_arg_index_ >= 0)` and \
                 `DT_CHECK(outer_loop_)` (AddressPinningAndToggle.cpp:2714-2715), with \
                 outer_loop_ {:?} and iter_arg_index_ {:?}",
                self.outer_loop,
                self.iter_arg_index
            )
        };
        let Some(init) = iter_operand(outer_loop, iter_arg_index, body) else {
            todo!(
                "getInit: {outer_loop:?} carries no value at {iter_arg_index:?} in this body, which \
                 is a loop and an index a valid ToggleDescriptor resolved together \
                 (AddressPinningAndToggle.cpp:2716-2717)"
            )
        };
        // `DT_CHECK(dcc::utils::isConstant<sentient::ConstantOp>(init))` — the op itself OR a
        // `uniform.query_map` all of whose mapped values are constants, never just the op.
        if !is_constant(init, ConstKind::ScalarConstant, defs) {
            todo!(
                "getInit: iter arg init {init:?} is expected to be constant in ToggleDescriptor \
                 (AddressPinningAndToggle.cpp:2718-2719)"
            )
        }
        todo!(
            "ExpressionEvaluator::evaluateValue (Analyses/ExpressionEvaluatorUtils, out of campaign \
             scope) on the constant iter arg init {init:?} that {outer_loop:?} starts \
             {iter_arg_index:?} at (AddressPinningAndToggle.cpp:2720-2721)"
        )
    }
}

/// `dcc::utils::getLoopNestLevel<sentient::ForOp>(outer_loop_)` (`dcc/src/Utils/Utils.cpp:211-220`) —
/// how many `sentient.for`s enclose this one, the outermost answering 0.
///
/// ⛔ ZERO-BASED BECAUSE THE WALK COUNTS THE LOOP ITSELF: `level` starts at `-1` and its first step
/// goes to the loop's own parent, so an unnested loop answers 0.
/// ⭐ FINDING THE LOOP IS THE DROPPABLE MECHANISM `getParentOfType` HAS FOR FREE; not finding it is
/// this island's version of `DT_CHECK_MSG(loop_op && isa<LoopTy>(loop_op), "expected valid loop")`
/// (`:213`).
fn loop_nest_level(outer_loop: ForRef, scope: &[Op]) -> i64 {
    /// The count of enclosing `sentient.for`s at the point `outer_loop` is found, if it is here.
    fn find(outer_loop: ForRef, scope: &[Op], enclosing: i64) -> Option<i64> {
        for op in scope {
            if let Op::Sentient(inner) = op {
                if let sentient::Op::For { iv, .. } = inner
                    && *iv == outer_loop.0
                {
                    return Some(enclosing);
                }
                // ⭐ ONLY A `sentient.for` COUNTS — `getParentOfType<LoopTy>` skips every other
                // enclosing op, `affine.for` included.
                let deeper = enclosing + i64::from(matches!(inner, sentient::Op::For { .. }));
                for region in sentient::regions(inner) {
                    if let Some(level) = find(outer_loop, region, deeper) {
                        return Some(level);
                    }
                }
            }
            if let Op::AffineFor(loop_op) = op
                && let Some(level) = find(outer_loop, &loop_op.body, enclosing)
            {
                return Some(level);
            }
        }
        None
    }
    match find(outer_loop, scope, 0) {
        Some(level) => level,
        None => todo!(
            "getLoopNestLevel<sentient::ForOp>: {outer_loop:?} is not in the body this descriptor \
             was dumped against, so the walk to its parents has nowhere to start \
             (dcc/src/Utils/Utils.cpp:213)"
        ),
    }
}

/// `outer_loop.getIterOperands()[iter_arg_index]` — the loop named by its induction variable, then
/// the initial value of the one carried entry that index picks out.
fn iter_operand(outer_loop: ForRef, iter_arg_index: IterArgIndex, scope: &[Op]) -> Option<Val> {
    for op in scope {
        if let Op::Sentient(inner) = op {
            if let sentient::Op::For { iv, carried, .. } = inner
                && *iv == outer_loop.0
            {
                return carried
                    .get(iter_arg_index.0 as usize)
                    .map(|entry| entry.init);
            }
            for region in sentient::regions(inner) {
                if let Some(init) = iter_operand(outer_loop, iter_arg_index, region) {
                    return Some(init);
                }
            }
        }
        if let Op::AffineFor(loop_op) = op
            && let Some(init) = iter_operand(outer_loop, iter_arg_index, &loop_op.body)
        {
            return Some(init);
        }
    }
    None
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::sentient::{Carried, Reg, RegType};
    use crate::transform::sentient::analyses::OutOfScopeEvaluator;

    /// `%base = sentient.scalar_sub %c, %arg` inside a loop that carries `%base` back — the shape
    /// e552 matches, with the pieces the ladder reads at the positions it reads them.
    fn toggle_body(yielded: Val) -> Vec<Op> {
        vec![
            Op::Sentient(sentient::Op::ScalarConstant {
                value: 4096,
                result: Val(1),
                reg_locale: RegType::Imm,
                ty: ScalarTy::Index,
                is_symbol: false,
            }),
            Op::Sentient(sentient::Op::For {
                iv: Val(2),
                bound: Val(0),
                bound_reg: None,
                carried: vec![Carried {
                    init: Val(1),
                    arg: Val(3),
                    result: Val(4),
                    reg: Reg {
                        locale: RegType::Lbr,
                        index: None,
                    },
                    program_header: false,
                    element_size: None,
                }],
                dbg_name: None,
                body: vec![
                    Op::Sentient(sentient::Op::ScalarSub {
                        lhs: Val(1),
                        rhs: Val(3),
                        result: Val(5),
                        reg: None,
                        element_size: None,
                        ty: ScalarTy::Index,
                    }),
                    Op::Sentient(sentient::Op::Yield {
                        results: vec![yielded],
                    }),
                ],
            }),
        ]
    }

    /// The resolution IS the ported half, so the seam it stops at names the value it resolved.
    #[test]
    #[should_panic(expected = "evaluateValue")]
    fn e015_resolves_the_iter_arg_to_its_constant_init() {
        let body = vec![
            Op::Sentient(sentient::Op::ScalarConstant {
                value: 4096,
                result: Val(1),
                reg_locale: RegType::Imm,
                ty: ScalarTy::Index,
                is_symbol: false,
            }),
            Op::Sentient(sentient::Op::For {
                iv: Val(2),
                bound: Val(0),
                bound_reg: None,
                carried: vec![Carried {
                    init: Val(1),
                    arg: Val(3),
                    result: Val(4),
                    reg: Reg {
                        locale: RegType::Lbr,
                        index: None,
                    },
                    program_header: false,
                    element_size: None,
                }],
                dbg_name: None,
                body: Vec::new(),
            }),
        ];
        let toggle = ToggleDescriptor {
            outer_loop: Some(ForRef(Val(2))),
            iter_arg_index: Some(IterArgIndex(0)),
            c1: None,
            can_be_simplified: false,
        };
        let regions: [&[Op]; 1] = [&body];
        let _evaluated = toggle.init(&body, Definitions::from_innermost(&regions));
    }

    /// 552/656 — the whole ladder passes on the vendor's shape and stops at the evaluator: the sub of
    /// a constant and a loop-carried arg, the loop's outermost constant initialization, the yield
    /// operand being the sub itself and its one other use.
    #[test]
    #[should_panic(expected = "evaluateValue")]
    fn e552_matches_a_sub_of_a_constant_and_the_iter_arg_the_loop_yields_back() {
        let body = toggle_body(Val(5));
        let regions: [&[Op]; 2] = [&body[1..], &body];
        let defs = Definitions::from_innermost(&regions);
        let _desc = ToggleDescriptor::new(Val(5), defs, &mut OutOfScopeEvaluator);
    }

    /// A loop that yields something ELSE back is `invalidate()`, not a toggle — and the fields the
    /// walk had already filled are cleared with it.
    #[test]
    fn e552_invalidates_when_the_loop_does_not_yield_the_base_address_back() {
        let body = toggle_body(Val(3));
        let regions: [&[Op]; 2] = [&body[1..], &body];
        let defs = Definitions::from_innermost(&regions);
        assert_eq!(
            ToggleDescriptor::new(Val(5), defs, &mut OutOfScopeEvaluator),
            ToggleDescriptor::default()
        );
    }

    /// 553/656 — an invalid toggle dumps its one indented line and nothing else, which is the whole
    /// branch that does not reach the out-of-scope `operator<<`.
    #[test]
    fn e553_dumps_invalid_and_stops() {
        let regions: [&[Op]; 1] = [&[]];
        let dumped = ToggleDescriptor::default().dump(
            &[],
            Definitions::from_innermost(&regions),
            &mut OutOfScopeEvaluator,
        );
        assert_eq!(dumped, "Toggle Descriptor:\n\tInvalid\n");
    }
}
