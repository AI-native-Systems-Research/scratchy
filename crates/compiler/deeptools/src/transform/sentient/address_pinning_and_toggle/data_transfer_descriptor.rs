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

//! `AddressPinningAndToggle.cpp` — 4 of the campaign's 656 units (dependency level(s) [1, 3, 5]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e278_isValid` | 278 | 1 | 23 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2478` |
//! | `e279_canBeSimplified` | 279 | 1 | 18 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2502` |
//! | `e492_dump` | 492 | 3 | 28 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2521` |
//! | `e593_initializeDescriptor` | 593 | 5 | 161 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2316` |

// crustify:todo: e492_dump
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2521  (28 body lines, level 3)
//   original  : void DataTransferDescriptor::dump() const
//   calls     : e258_isToggle, e259_isConditionalConstant, e260_isIntegerSequence, e261_isDiscreteIntegerSet, e262_isLoopingChainMutableAddr, e415_isSimpleConstant

// crustify:todo: e593_initializeDescriptor
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2316  (161 body lines, level 5)
//   original  : void DataTransferDescriptor::initializeDescriptor()
//   calls     : e002_getAllConstants, e015_getInit, e016_ConditionalConstantDescriptor, e252_size, e278_isValid, e279_canBeSimplified, e280_IntegerSequenceDescriptor, e281_DiscreteIntegerSetDescriptor, e282_LoopingChainMutableAddrDescriptor, e407_getInit, e408_getAllConstants, e411_getInit, e414_getInit, e485_getX …

use super::{BaseAddrList, PatternDescriptor};

/// ONE DATA TRANSFER'S BASE-ADDRESS STORY — `class DataTransferDescriptor`
/// (`AddressPinningAndToggle.cpp:632-790`), one per `load_and_send`/`receive_and_store`/
/// `load_and_store` per address role (the HBM `load_and_store` case makes TWO, `:1410-1415`).
///
/// ⛔ NOT `evaluator_`: the reference stores `ExpressionEvaluator &`, one global the pass threads
/// through. A shared borrow in a field would make the descriptor unstorable while the pass rewrites
/// the IR, so the ported methods take the evaluator as an argument instead.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DataTransferDescriptor {
    /// Replaces: e006_dtor_DataTransferDescriptor
    ///
    /// `pattern_desc_` — which recognised pattern this transfer follows, `None` for the reference's
    /// `nullptr` (every `is*()` tests it, `:673-696`).
    ///
    /// ⭐ THIS FIELD *IS* THE PORT OF `~DataTransferDescriptor()` (`:645-647`, whose whole body is
    /// `if (pattern_desc_) delete pattern_desc_;`): the destructor exists only because the reference
    /// holds `DynamicPatternDescriptorBase *` from a `new` in `initializeDescriptor`. An owned
    /// `Option` frees exactly that, at exactly that point, so ⛔ there is no `impl Drop` to write —
    /// a hand-written one restating the field's own drop would free nothing extra.
    pub pattern_desc: Option<PatternDescriptor>,
    /// `base_addrs_` — the possible constant values `base_addr_` can take (one, or two for a
    /// toggle); the list [`super::ConditionalConstantDescriptor::get_all_constants`] appends into.
    pub base_addrs: BaseAddrList,
}

impl DataTransferDescriptor {
    /// Replaces: e278_isValid
    ///
    /// Whether this transfer follows a recognised pattern: no base addresses is invalid, a matched
    /// pattern answers for itself, and anything else needs exactly one base address (`:2478-2500`).
    ///
    /// ⛔ THE TOGGLE ARM IS THE ONLY ONE THAT COUNTS BASE ADDRESSES (`:2483-2486`): two of them, or
    /// one when the toggle simplified away.
    /// ⛔ THERE IS NO `SimpleConstant` ARM IN THE REFERENCE'S `dyn_cast` CHAIN — a simple constant
    /// falls through to the `base_addrs.size() == 1` tail, exactly as `pattern_desc_ == nullptr` does.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        if self.base_addrs.is_empty() {
            return false;
        }
        match &self.pattern_desc {
            Some(PatternDescriptor::Toggle(toggle)) => {
                toggle.is_valid()
                    && ((toggle.can_be_simplified && self.base_addrs.len() == 1)
                        || self.base_addrs.len() == 2)
            }
            Some(PatternDescriptor::ConditionalConstant(cc)) => cc.is_valid(),
            Some(PatternDescriptor::IntegerSequence(isq)) => isq.is_valid(),
            Some(PatternDescriptor::DiscreteIntegerSet(dis)) => dis.is_valid(),
            Some(PatternDescriptor::LoopingChainMutableAddr(lcma)) => lcma.is_valid(),
            Some(PatternDescriptor::SimpleConstant(_)) | None => self.base_addrs.len() == 1,
        }
    }

    /// Replaces: e279_canBeSimplified
    ///
    /// Whether the matched pattern collapsed to a single constant base address — the pattern's own
    /// `can_be_simplified_`, and `false` for no pattern at all (`:2502-2519`).
    ///
    /// ⛔ EVERY `is*()` OPENS WITH `isValid()` (`:673-696`), so an invalid descriptor answers `false`
    /// however its pattern's own flag stands.
    /// ⛔ A SIMPLE CONSTANT ANSWERS `false` DESPITE ITS OWN `setCanBeSimplified(true)` (`:169`):
    /// `isSimpleConstant()` is absent from this `else if` chain, and nothing reads that flag here.
    #[must_use]
    pub fn can_be_simplified(&self) -> bool {
        let res = self.is_valid()
            && match &self.pattern_desc {
                Some(PatternDescriptor::Toggle(toggle)) => toggle.can_be_simplified,
                Some(PatternDescriptor::ConditionalConstant(cc)) => cc.can_be_simplified,
                Some(PatternDescriptor::IntegerSequence(isq)) => isq.can_be_simplified,
                Some(PatternDescriptor::DiscreteIntegerSet(dis)) => dis.can_be_simplified,
                Some(PatternDescriptor::LoopingChainMutableAddr(lcma)) => lcma.can_be_simplified,
                Some(PatternDescriptor::SimpleConstant(_)) | None => false,
            };
        // `DT_CHECK_MSG((!res || getBaseAddrList().size() == 1), ..)` (`:2514-2518`) — an ABORT in the
        // reference, so it stays a named stop rather than becoming a refusal.
        if res && self.base_addrs.len() != 1 {
            todo!(
                "DataTransferDescriptor::canBeSimplified: DT_CHECK_MSG(!res || \
                 getBaseAddrList().size() == 1, \"simplified pattern should have a single \
                 base_addr stored in the descriptor\") — {} base addrs (:2514-2518)",
                self.base_addrs.len()
            )
        }
        res
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::islands::sentient::dialects::Val;
    use crate::transform::sentient::address_pinning_and_toggle::{
        SimpleConstantDescriptor, ToggleDescriptor,
    };
    use crate::transform::sentient::analyses::EvaluatedValue;
    use crate::transform::sentient::{ForRef, IterArgIndex};

    /// A toggle that matched, so only the base-address count decides.
    fn matched_toggle(can_be_simplified: bool) -> ToggleDescriptor {
        ToggleDescriptor {
            outer_loop: Some(ForRef(Val(1))),
            iter_arg_index: Some(IterArgIndex(0)),
            c1: Some(EvaluatedValue(2)),
            can_be_simplified,
        }
    }

    fn descriptor(pattern: Option<PatternDescriptor>, base_addrs: u32) -> DataTransferDescriptor {
        DataTransferDescriptor {
            pattern_desc: pattern,
            base_addrs: (0..base_addrs).map(EvaluatedValue).collect(),
        }
    }

    fn simple_constant() -> PatternDescriptor {
        PatternDescriptor::SimpleConstant(SimpleConstantDescriptor {
            ev: EvaluatedValue(2),
        })
    }

    /// The toggle arm's base-address count, and the `SimpleConstant` the `dyn_cast` chain skips.
    #[test]
    fn e278_counts_base_addrs_for_a_toggle_and_falls_through_for_a_simple_constant() {
        let toggle = |simplified, count| {
            descriptor(
                Some(PatternDescriptor::Toggle(matched_toggle(simplified))),
                count,
            )
            .is_valid()
        };
        assert!(toggle(false, 2));
        assert!(!toggle(false, 1));
        assert!(toggle(true, 1));
        // ⭐ `|| base_addrs.size() == 2` IS NOT GUARDED BY THE FLAG, so a simplified toggle holding
        // two is valid too.
        assert!(toggle(true, 2));
        // No `SimpleConstant` arm: it reaches the `base_addrs.size() == 1` tail, as `None` does.
        assert!(descriptor(Some(simple_constant()), 1).is_valid());
        assert!(!descriptor(Some(simple_constant()), 2).is_valid());
        assert!(descriptor(None, 1).is_valid());
        assert!(!DataTransferDescriptor::default().is_valid());
    }

    /// The pattern's own flag, gated on validity — and the simple constant whose `true` is unread.
    #[test]
    fn e279_reads_the_patterns_flag_and_never_the_simple_constants() {
        assert!(
            descriptor(Some(PatternDescriptor::Toggle(matched_toggle(true))), 1)
                .can_be_simplified()
        );
        // Every `is*()` opens with `isValid()`, which an unmatched toggle fails.
        assert!(
            !descriptor(
                Some(PatternDescriptor::Toggle(ToggleDescriptor::default())),
                1
            )
            .can_be_simplified()
        );
        assert!(!descriptor(Some(simple_constant()), 1).can_be_simplified());
    }

    /// `DT_CHECK_MSG((!res || getBaseAddrList().size() == 1), ..)` is an abort, and a simplified
    /// toggle holding two base addresses is a descriptor that reaches it.
    #[test]
    #[should_panic(expected = "single base_addr stored in the descriptor")]
    fn e279_aborts_on_a_simplified_pattern_with_more_than_one_base_addr() {
        let desc = descriptor(Some(PatternDescriptor::Toggle(matched_toggle(true))), 2);
        let _simplified = desc.can_be_simplified();
    }
}
