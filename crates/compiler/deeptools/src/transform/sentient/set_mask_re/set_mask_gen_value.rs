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

//! `SetMaskRE.cpp` — 4 of the campaign's 656 units (dependency level(s) [0, 1]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e187_copyTo` | 187 | 0 | 6 | `dcc/src/Transform/Sentient/SetMaskRE.cpp:243` |
//! | `e188_print` | 188 | 0 | 5 | `dcc/src/Transform/Sentient/SetMaskRE.cpp:252` |
//! | `e189_maskValuesAreEquivalent` | 189 | 0 | 13 | `dcc/src/Transform/Sentient/SetMaskRE.cpp:258` |
//! | `e376_isEqual` | 376 | 1 | 7 | `dcc/src/Transform/Sentient/SetMaskRE.cpp:235` |

// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET, so every item below is reachable only from this
// module's own tests until `e474_runOn`/`e536_runOnOperation` (levels 2/3) land and something calls
// it. CI runs clippy with `-D warnings`, so without this the first ported leaf fails the gate.
// ⭐ REMOVE THIS WITH e474: at that point an unused item here is a real defect again.
#![allow(dead_code)]

use crate::islands::dataflow_ir::print;
use crate::islands::sentient::dialects::{Definitions, Op, Val, sentient};
use crate::transform::sentient::cfg_simplification_sentient_level::pattern_simplification_manager::OpPath;

/// `SetMaskGenValue` (`SetMaskRE.hpp:25`) — the definition a `sentient.set_mask` node generates.
///
/// ⭐ DECLARED HERE, WHERE ITS OWN METHODS BELONG: e187-e189 and e376 are `copyTo`/`print`/
/// `maskValuesAreEquivalent`/`isEqual` on this class and are scheduled into this file. e375 (in
/// [`super::set_mask_rde_tree`]) constructs it — a batch filling that anchor should UNION with this.
#[derive(Debug, Clone, Default)]
pub(crate) struct SetMaskGenValue {
    /// `mask_value_` — `None` is the constructor's `nullptr`, i.e. `isUnknownValue()`.
    mask_value: Option<Val>,
    /// `DataFlowDefinitionBase::op_`, which e187 copies and e194 rewrites through.
    ///
    /// ⛔ AN [`OpPath`] AND NOT THE NODE'S OWN OP: `node_gen.copyTo(parent_gen)`
    /// (`Analyses/RedundantDefinitionEliminationTreeImpl.cpp:355`) hands a child's `op_` to its
    /// PARENT, so a gen's operation may sit in a nested block and needs an identity of its own.
    op: Option<OpPath>,
    /// `DataFlowDefinitionBase::is_optimized_` — the base class is OUT OF CAMPAIGN SCOPE, but e188
    /// prints this flag, so the subclass holds it exactly as it already holds `op_`.
    is_optimized: bool,
    /// `DataFlowDefinitionBase::is_dead_`, printed by e188 and set by e193.
    is_dead: bool,
}

impl SetMaskGenValue {
    /// `SetMaskGenValue()` — the unknown value.
    #[must_use]
    pub(crate) fn unknown() -> SetMaskGenValue {
        SetMaskGenValue::default()
    }

    /// `SetMaskGenValue(mask_value, op)`.
    #[must_use]
    pub(crate) fn of(mask_value: Val, op: OpPath) -> SetMaskGenValue {
        SetMaskGenValue {
            mask_value: Some(mask_value),
            op: Some(op),
            is_optimized: false,
            is_dead: false,
        }
    }

    /// `getMaskValue()`, absent when the value is unknown.
    #[must_use]
    pub(crate) const fn mask_value(&self) -> Option<Val> {
        self.mask_value
    }

    /// `setMaskValue(v)`.
    pub(crate) const fn set_mask_value(&mut self, mask_value: Val) {
        self.mask_value = Some(mask_value);
    }

    /// `getOperation()` — the `sentient.set_mask` this definition came from.
    #[must_use]
    pub(crate) const fn op(&self) -> Option<&OpPath> {
        self.op.as_ref()
    }

    /// `isUnknownValue()` (`SetMaskRE.cpp:250`).
    #[must_use]
    pub(crate) const fn is_unknown_value(&self) -> bool {
        self.mask_value.is_none()
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

    /// `DataFlowDefinitionBase::setIsDead()`.
    pub(crate) const fn set_is_dead(&mut self) {
        self.is_dead = true;
    }

    /// Replaces: e187_copyTo
    ///
    /// Copies the mask value and the generating operation onto `lhs`, leaving its two flags alone.
    ///
    /// ⛔ THE `assert(lhs_p && "invalid subclasses of dataflow definitions")` IS THE PARAMETER TYPE:
    /// a definition that is not a `SetMaskGenValue` is not expressible at this call.
    ///
    /// ⛔ AND `is_optimized_`/`is_dead_` STAY BEHIND — the reference writes two fields, so
    /// `*lhs = self.clone()` would clobber both.
    pub(crate) fn copy_to(&self, lhs: &mut SetMaskGenValue) {
        lhs.mask_value = self.mask_value;
        lhs.op = self.op.clone();
    }

    /// Replaces: e188_print
    ///
    /// Renders the GenValue as the pass's `-debug-only=setmask-re` dump does.
    ///
    /// ⛔ DEVIATION, AND IT IS OBSERVABLE: `mask_value_.getAsOpaquePointer()` streams the HOST
    /// ADDRESS of the `Value`'s implementation, which this island has no equivalent of — so the dump
    /// names the value, `%N`, and keeps `0x0` for the null the pointer form did say.
    pub(crate) fn print(&self, out: &mut String) {
        out.push_str("(GenValue: value_:");
        match self.mask_value {
            Some(mask_value) => out.push_str(&print::val(mask_value)),
            None => out.push_str("0x0"),
        }
        out.push(')');
        if self.is_optimized {
            out.push_str(" - optimized!");
        }
        if self.is_dead {
            out.push_str(" - dead!");
        }
    }

    /// Replaces: e189_maskValuesAreEquivalent
    ///
    /// Two mask values are equivalent when they are the same value, or both are
    /// `sentient.scalar_constant`s carrying the same number.
    ///
    /// ⛔ TWO ABSENT VALUES ARE EQUIVALENT AND ONE IS NOT: `a == b` runs FIRST, so null/null is
    /// `true` before the `!a || !b` rejection ever sees it.
    ///
    /// ⭐ AN ASSOCIATED FUNCTION: the body reads no member, and `defs` is the scope MLIR's own
    /// `Value::getDefiningOp()` needs no parameter for.
    #[must_use]
    pub(crate) fn mask_values_are_equivalent(
        a: Option<Val>,
        b: Option<Val>,
        defs: Definitions<'_>,
    ) -> bool {
        if a == b {
            return true;
        }
        let (Some(a), Some(b)) = (a, b) else {
            return false;
        };
        match (defs.of(a), defs.of(b)) {
            (
                Some(Op::Sentient(sentient::Op::ScalarConstant { value: a_value, .. })),
                Some(Op::Sentient(sentient::Op::ScalarConstant { value: b_value, .. })),
            ) => a_value == b_value,
            _ => false,
        }
    }
}

// crustify:todo: e376_isEqual
//   authority : dcc/src/Transform/Sentient/SetMaskRE.cpp:235  (7 body lines, level 1)
//   original  : bool SetMaskGenValue::isEqual(const DataFlowDefinitionBase &rhs) const
//   calls     : e189_maskValuesAreEquivalent

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::islands::dataflow_ir::ty::ScalarTy;

    /// `%result = sentient.scalar_constant {value = <value>}`.
    fn constant(result: u32, value: i64) -> Op {
        Op::Sentient(sentient::Op::ScalarConstant {
            value,
            result: Val(result),
            reg_locale: sentient::RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    /// A GenValue of `mask_value` with both base flags set.
    fn flagged(value: SetMaskGenValue) -> SetMaskGenValue {
        SetMaskGenValue {
            is_optimized: true,
            is_dead: true,
            ..value
        }
    }

    /// e187 — the mask value and `op_` travel; `is_optimized_` and `is_dead_` stay behind.
    #[test]
    fn copy_to_moves_the_mask_value_and_op_but_not_the_flags() {
        let source = flagged(SetMaskGenValue::of(Val(7), OpPath::at(&[(0, 3)])));
        let mut target = SetMaskGenValue::unknown();

        source.copy_to(&mut target);

        assert_eq!(target.mask_value(), Some(Val(7)));
        assert_eq!(target.op(), Some(&OpPath::at(&[(0, 3)])));
        assert!(!target.is_optimized(), "is_optimized_ is not copied");
        assert!(!target.is_dead(), "is_dead_ is not copied");
    }

    /// e188 — the null mask value prints as `0x0`, and each flag adds its own suffix.
    #[test]
    fn print_names_the_value_and_both_suffixes() {
        let mut out = String::new();
        flagged(SetMaskGenValue::unknown()).print(&mut out);
        assert_eq!(out, "(GenValue: value_:0x0) - optimized! - dead!");

        let mut plain = String::new();
        SetMaskGenValue::of(Val(7), OpPath::at(&[(0, 3)])).print(&mut plain);
        assert_eq!(plain, "(GenValue: value_:%7)");
    }

    /// e189 — one value, two equal constants, and the three ways to be inequivalent.
    #[test]
    fn equivalent_mask_values_are_the_same_value_or_two_equal_constants() {
        let scope = vec![
            constant(1, 4),
            constant(2, 4),
            constant(3, 5),
            Op::Sentient(sentient::Op::Nop { dbg_name: None }),
        ];
        let module = [scope.as_slice()];
        let defs = Definitions::from_innermost(&module);

        let equivalent = |a, b| SetMaskGenValue::mask_values_are_equivalent(a, b, defs);
        assert!(equivalent(Some(Val(1)), Some(Val(1))), "the same value");
        assert!(equivalent(None, None), "`a == b` runs before the null test");
        assert!(equivalent(Some(Val(1)), Some(Val(2))), "two constants of 4");
        assert!(!equivalent(Some(Val(1)), Some(Val(3))), "4 is not 5");
        assert!(!equivalent(Some(Val(1)), None), "one null value");
        assert!(
            !equivalent(Some(Val(1)), Some(Val(9))),
            "nothing defines %9"
        );
    }
}
