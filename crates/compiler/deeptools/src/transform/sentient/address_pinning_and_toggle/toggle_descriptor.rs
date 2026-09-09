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

// crustify:todo: e552_ToggleDescriptor
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2642  (68 body lines, level 4)
//   original  : ToggleDescriptor::ToggleDescriptor(ExpressionEvaluator &evaluator, Value base_addr) : DynamicPatternDescriptorBase(PatternKind::kToggle, evaluator)
//   calls     : e001_invalidate, e003_invalidate, e004_invalidate, e005_invalidate, e017_getOutermostConstInitialization, e485_getX

// crustify:todo: e553_dump
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2724  (16 body lines, level 4)
//   original  : void ToggleDescriptor::dump() const
//   calls     : e278_isValid, e279_canBeSimplified, e401_getC1, e485_getX

use crate::islands::sentient::dialects::{Definitions, Op, Val, sentient};
use crate::transform::sentient::analyses::EvaluatedValue;
use crate::transform::sentient::utils::{ConstKind, is_constant};
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
                iv_reg: sentient::Reg::UNALLOCATED,
                iv: Val(2),
                bound: Val(0),
                carried: vec![Carried {
                    result_reg: Reg::UNALLOCATED,
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
}
