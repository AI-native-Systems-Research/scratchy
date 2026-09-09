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

//! `SetMaskRE.cpp` — 3 of the campaign's 656 units (dependency level(s) [0]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e190_isEqual` | 190 | 0 | 5 | `dcc/src/Transform/Sentient/SetMaskRE.cpp:276` |
//! | `e191_copyTo` | 191 | 0 | 5 | `dcc/src/Transform/Sentient/SetMaskRE.cpp:282` |
//! | `e192_print` | 192 | 0 | 5 | `dcc/src/Transform/Sentient/SetMaskRE.cpp:290` |

// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET, so `IncrMaskGenValue` is reachable only from this
// file's own tests until `e536_runOnOperation` (level 3) lands and something calls it. CI runs clippy
// with `-D warnings`, so without this the first ported leaf of the module fails the gate.
// ⭐ REMOVE THIS WITH e536: at that point an unused item here is a real defect again.
#![allow(dead_code)]

use crate::islands::sentient::dialects::{Op, sentient};
use crate::transform::sentient::cfg_simplification_sentient_level::pattern_simplification_manager::OpPath;

/// HOW MUCH A `sentient.incrmask` INCREMENTS THE MASK — a witness, because the answer is a constant.
///
/// ⛔ `int getIncrement() { return 1; }` (`SentientOps.td:1081`) — the op has NO operands and no
/// increment attribute, so ONE is the only value `IncrMaskGenValue::increment_` can ever hold and
/// nothing in the tree calls `setIncrement`. A field would invite a second answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Increment;

impl Increment {
    /// `getIncrement()`.
    #[must_use]
    pub(crate) const fn get(self) -> i32 {
        1
    }
}

/// A TARGET FOR `IncrMaskGenValue::copyTo` — uninhabited, because there is none.
///
/// ⛔⛔ THIS IS `llvm_unreachable("should never be attempting to copy an IncrMaskGenValue")`
/// (`SetMaskRE.cpp:285`) TURNED INTO A COMPILE ERROR. The reference's whole body is that abort; with
/// no value of this type constructible, the call it aborts on cannot be written at all, so the
/// guarantee holds at build time instead of costing a run.
#[derive(Debug, Clone, Copy)]
pub(crate) enum NoCopyTarget {}

/// `IncrMaskGenValue` (`SetMaskRE.hpp:49`) — the dataflow definition an RDE node generates for a
/// `sentient.incrmask`.
///
/// ⭐ DECLARED HERE, WHERE ITS OWN METHODS BELONG: e190-e192 are its `isEqual`/`copyTo`/`print` and
/// all three are scheduled into this file.
#[derive(Debug, Clone, Default)]
pub(crate) struct IncrMaskGenValue {
    /// `increment_` — `None` is the constructor's `-1`, i.e. `isUnknownValue()`.
    increment: Option<Increment>,
    /// `op_` — absent for the default-constructed unknown value.
    ///
    /// ⛔ A PATH, NOT AN OP: e193 and e194 push this into `to_be_deleted_` to be ERASED and rewrite
    /// the block around it, and two `sentient.incrmask` ops are indistinguishable by value — so the
    /// identity has to be the position, which is what [`OpPath`] is the stand-in for.
    op: Option<OpPath>,
    /// `DataFlowDefinitionBase::is_optimized_`
    /// (`Analyses/RedundantDefinitionEliminationTree.hpp:290`) — the base class is OUT OF CAMPAIGN
    /// SCOPE, but e192 prints this flag, so the subclass holds it exactly as it holds `op_`.
    is_optimized: bool,
    /// `DataFlowDefinitionBase::is_dead_`, printed by e192 for the same reason.
    is_dead: bool,
}

impl IncrMaskGenValue {
    /// `IncrMaskGenValue()` — the unknown value.
    #[must_use]
    pub(crate) fn unknown() -> IncrMaskGenValue {
        IncrMaskGenValue::default()
    }

    /// `IncrMaskGenValue(incrmask_op.getIncrement(), op)`, and nothing for any other operation.
    #[must_use]
    pub(crate) fn of_incr_mask(op: &Op, at: OpPath) -> Option<IncrMaskGenValue> {
        if !matches!(op, Op::Sentient(sentient::Op::IncrMask { .. })) {
            return None;
        }
        Some(IncrMaskGenValue {
            increment: Some(Increment),
            op: Some(at),
            is_optimized: false,
            is_dead: false,
        })
    }

    /// `isUnknownValue()` (`SetMaskRE.cpp:288`) — `increment_ < 0`.
    #[must_use]
    pub(crate) const fn is_unknown_value(&self) -> bool {
        self.increment.is_none()
    }

    /// `getIncrement()`, absent when the value is unknown.
    #[must_use]
    pub(crate) const fn increment(&self) -> Option<Increment> {
        self.increment
    }

    /// The op that generated it.
    #[must_use]
    pub(crate) const fn op(&self) -> Option<&OpPath> {
        self.op.as_ref()
    }

    /// `DataFlowDefinitionBase::isOptimized()`.
    #[must_use]
    pub(crate) const fn is_optimized(&self) -> bool {
        self.is_optimized
    }

    /// `DataFlowDefinitionBase::isDead()`.
    #[must_use]
    pub(crate) const fn is_dead(&self) -> bool {
        self.is_dead
    }

    /// `DataFlowDefinitionBase::setIsDead()` — one way, as the base class has no clearing setter.
    pub(crate) const fn set_is_dead(&mut self) {
        self.is_dead = true;
    }
}

impl IncrMaskGenValue {
    /// Replaces: e190_isEqual
    ///
    /// No two incrmask definitions are ever equal.
    ///
    /// ⛔ A CONSTANT `false` ON PURPOSE, NOT A STUB: the reference's comment says every incrmask is a
    /// unique invocation whose placement must not change, so the RDE tree must never common two.
    /// This is also why the type does not derive `PartialEq` — a second, disagreeing `==`.
    ///
    /// ⛔ IT IGNORES `rhs` ENTIRELY, including its own identity: `value.is_equal(&value)` is `false`.
    #[must_use]
    pub(crate) fn is_equal(&self, _rhs: &IncrMaskGenValue) -> bool {
        false
    }

    /// Replaces: e191_copyTo
    ///
    /// ⛔⛔ NOT AN OPERATION — the reference's body is `llvm_unreachable("should never be attempting to
    /// copy an IncrMaskGenValue")` (`SetMaskRE.cpp:282-286`), because an incrmask never moves.
    ///
    /// ⭐ THE ABORT IS THE PARAMETER TYPE: [`NoCopyTarget`] is uninhabited, so no caller can build the
    /// argument and the path the reference aborts on does not compile. The `match` has no arms for
    /// the same reason.
    pub(crate) fn copy_to(&self, lhs: NoCopyTarget) -> ! {
        match lhs {}
    }

    /// Replaces: e192_print
    ///
    /// Renders the GenValue as the pass's `-debug-only=set-mask-re` dump does.
    ///
    /// ⛔ AN ABSENT INCREMENT PRINTS `-1` — the reference streams the sentinel `int`, so the dump says
    /// `(GenValue: -1)` and never a word like "none".
    pub(crate) fn print(&self, out: &mut String) {
        out.push_str("(GenValue: ");
        match self.increment {
            Some(increment) => out.push_str(&increment.get().to_string()),
            None => out.push_str("-1"),
        }
        out.push(')');
        if self.is_optimized {
            out.push_str(" - optimized!");
        }
        if self.is_dead {
            out.push_str(" - dead!");
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::{IncrMaskGenValue, Increment, OpPath};
    use crate::islands::sentient::dialects::{Op, sentient};

    /// `sentient.incrmask`.
    fn incr_mask() -> Op {
        Op::Sentient(sentient::Op::IncrMask { dbg_name: None })
    }

    /// A GenValue with both base flags set.
    fn flagged(value: IncrMaskGenValue) -> IncrMaskGenValue {
        IncrMaskGenValue {
            is_optimized: true,
            is_dead: true,
            ..value
        }
    }

    /// e190 — nothing is equal to an incrmask definition, not even itself.
    ///
    /// ⭐ e191 NEEDS NO TEST AND CAN HAVE NONE: its argument type is uninhabited, so a call is a
    /// compile error rather than a run-time abort.
    #[test]
    fn no_incrmask_definition_is_ever_equal_to_another() {
        let value = IncrMaskGenValue::of_incr_mask(&incr_mask(), OpPath::at(&[(0, 0)]))
            .expect("an incrmask generates one");
        assert!(!value.is_equal(&value), "not even to itself");
        assert!(!value.is_equal(&IncrMaskGenValue::unknown()));
        assert!(!IncrMaskGenValue::unknown().is_equal(&IncrMaskGenValue::unknown()));
    }

    /// e192 — the sentinel prints as `-1`, a real increment as the constant `1`, and each flag adds
    /// its own suffix.
    #[test]
    fn print_renders_the_sentinel_as_minus_one_and_both_suffixes() {
        let mut out = String::new();
        flagged(IncrMaskGenValue::unknown()).print(&mut out);
        assert_eq!(out, "(GenValue: -1) - optimized! - dead!");

        let mut plain = String::new();
        IncrMaskGenValue::of_incr_mask(&incr_mask(), OpPath::at(&[(0, 0)]))
            .expect("an incrmask generates one")
            .print(&mut plain);
        assert_eq!(plain, "(GenValue: 1)");
        assert_eq!(Increment.get(), 1);
        assert!(
            IncrMaskGenValue::of_incr_mask(
                &Op::Sentient(sentient::Op::Nop { dbg_name: None }),
                OpPath::at(&[(0, 0)])
            )
            .is_none()
        );
    }
}
