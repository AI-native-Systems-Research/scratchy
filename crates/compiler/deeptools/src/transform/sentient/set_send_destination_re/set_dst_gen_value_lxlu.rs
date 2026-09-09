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

//! `SetSendDestinationRE.cpp` — 3 of the campaign's 656 units (dependency level(s) [0]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e198_isEqual` | 198 | 0 | 24 | `dcc/src/Transform/Sentient/SetSendDestinationRE.cpp:322` |
//! | `e199_copyTo` | 199 | 0 | 22 | `dcc/src/Transform/Sentient/SetSendDestinationRE.cpp:347` |
//! | `e200_print` | 200 | 0 | 19 | `dcc/src/Transform/Sentient/SetSendDestinationRE.cpp:370` |

use super::QueryMapOp;
use super::composite_set_dst_gen_value_lxlu::CompositeSetDstGenValueLxlu;
use super::simple_set_dst_gen_value_lxlu::{SendDestination, SimpleSetDstGenValueLxlu};
use crate::islands::sentient::dialects::Op;

/// `SetDstGenValueLXLU::Kind` AND ITS `union GenValue` AS ONE VALUE
/// (`SetSendDestinationRE.hpp:82-86,114-123`).
///
/// ⛔⛔ THE TAG AND THE UNION ARE ONE THING, WHICH IS WHAT MAKES `llvm_unreachable("unhandled case")`
/// AND `llvm_unreachable("kind of gen value not fully implemented")` UNREPRESENTABLE. In the
/// reference `kind_` and `gen_value_` can disagree — `SetDstGenValueLXLU(Kind::kSimple, op)` sets the
/// tag and leaves the union member unconstructed — and the two `llvm_unreachable`s are what happens
/// when they do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum GenValue {
    /// `kUnknown` — the most pessimistic mode, and the default-constructed value.
    #[default]
    Unknown,
    /// `kSimple` — the destination is a `dataflow.get_unit`.
    Simple(SimpleSetDstGenValueLxlu),
    /// `kComposite` — the destination comes out of a `uniform.query_map`.
    Composite(CompositeSetDstGenValueLxlu),
}

/// `SetDstGenValueLXLU` (`SetSendDestinationRE.hpp:80`) — the dataflow definition an RDE node
/// generates when this pass is optimizing the LXLU's `SETDSTMASK`.
///
/// ⭐ THE BASE CLASS'S THREE FIELDS ARE HELD HERE. `DataFlowDefinitionBase`
/// (`Analyses/RedundantDefinitionEliminationTree.hpp:327`) is OUT OF CAMPAIGN SCOPE, and e199 writes
/// `op_` while e200 prints both flags — so the subclass carries them, exactly as
/// [`crate::transform::sentient::implicit_sync_re::ImplicitSyncGenValue`] does.
#[derive(Debug, Clone, Default)]
pub(crate) struct SetDstGenValueLxlu {
    gen_value: GenValue,
    /// `op_` — absent for the default-constructed unknown value.
    op: Option<Op>,
    /// `is_optimized_`; nothing in scope sets it yet.
    is_optimized: bool,
    /// `is_dead_`.
    is_dead: bool,
}

impl SetDstGenValueLxlu {
    /// `SetDstGenValueLXLU()` — the unknown value.
    #[must_use]
    pub(crate) fn unknown() -> SetDstGenValueLxlu {
        SetDstGenValueLxlu::default()
    }

    /// `SetDstGenValueLXLU(mode, op)` (`:95`).
    #[must_use]
    pub(crate) fn simple(value: SendDestination, op: Op) -> SetDstGenValueLxlu {
        SetDstGenValueLxlu {
            gen_value: GenValue::Simple(SimpleSetDstGenValueLxlu::of(value)),
            op: Some(op),
            is_optimized: false,
            is_dead: false,
        }
    }

    /// `SetDstGenValueLXLU(qmap, op)` (`:90`).
    #[must_use]
    pub(crate) fn composite(qmap: QueryMapOp, op: Op) -> SetDstGenValueLxlu {
        SetDstGenValueLxlu {
            gen_value: GenValue::Composite(CompositeSetDstGenValueLxlu::of(qmap)),
            op: Some(op),
            is_optimized: false,
            is_dead: false,
        }
    }

    /// `isUnknownValue()` — `!isSimple() && !isComposite()` (`:107-109`).
    #[must_use]
    pub(crate) const fn is_unknown_value(&self) -> bool {
        matches!(self.gen_value, GenValue::Unknown)
    }

    /// What it generates.
    #[must_use]
    pub(crate) const fn gen_value(&self) -> GenValue {
        self.gen_value
    }

    /// `getOperation()`.
    #[must_use]
    pub(crate) const fn op(&self) -> Option<&Op> {
        self.op.as_ref()
    }

    /// Replaces: e198_isEqual
    ///
    /// Two LXLU GenValues are equal when both are known, of the same kind, and agree on that kind's
    /// value — the send destination, or the query map by identity.
    ///
    /// ⛔ AN UNKNOWN IS EQUAL TO NOTHING, INCLUDING ANOTHER UNKNOWN: `if (isUnknownValue() ||
    /// rhs.isUnknownValue()) return false` (`SetSendDestinationRE.cpp:323`). ⭐ NOTE THIS DIVERGES
    /// FROM THE IMPLICIT-SYNC TWIN, where `-1 == -1` holds.
    ///
    /// ⛔ THE `is_optimized` BIT IS INTENTIONALLY LEFT OUT — the reference says so in as many words —
    /// and so are `is_dead_` and `op_`. This is also why the type derives no `PartialEq`.
    ///
    /// ⛔ THE `dynamic_cast` AND ITS `DT_CHECK_MSG` BECAME THE PARAMETER TYPE, and the cross-kind
    /// `return false` pair stays because it is a real answer, not a check.
    #[must_use]
    pub(crate) fn is_equal(&self, rhs: &SetDstGenValueLxlu) -> bool {
        match (self.gen_value, rhs.gen_value) {
            (GenValue::Unknown, _) | (_, GenValue::Unknown) => false,
            (GenValue::Simple(lhs), GenValue::Simple(rhs)) => lhs.value() == rhs.value(),
            // "todo: for now we establish equality based on shallow SSA value comparison" (`:338-340`).
            (GenValue::Composite(lhs), GenValue::Composite(rhs)) => {
                lhs.query_map() == rhs.query_map()
            }
            // "We don't have valid scenarios where we need to optimize across simple and composite
            // nodes" (`:328-329`).
            (GenValue::Simple(_), GenValue::Composite(_))
            | (GenValue::Composite(_), GenValue::Simple(_)) => false,
        }
    }

    /// Replaces: e199_copyTo
    ///
    /// Copies the generating operation and the generated value onto `lhs`, leaving its two flags
    /// alone.
    ///
    /// ⛔ "COPY EVERYTHING EXCEPT THE `is_optimized_` FLAG" — and except `is_dead_`, which the
    /// reference also never assigns here. `*lhs = self.clone()` would clobber both.
    ///
    /// ⛔⛔ `DT_CHECK_MSG(lhs_p->kind_ == Kind::kUnknown, "mixed mode copying is currently only
    /// supported when copying into an unknown")` IS A UNION-SAFETY CHECK, AND THE TAGGED
    /// [`GenValue`] DISCHARGES IT. The reference can only retag `lhs` while its union member is dead,
    /// because writing the other member of a live union is what it has no way to do safely; assigning
    /// one Rust enum value replaces tag and payload together, so every kind pair lands on the same
    /// result the reference produces for the pairs it permits.
    pub(crate) fn copy_to(&self, lhs: &mut SetDstGenValueLxlu) {
        lhs.op = self.op.clone();
        lhs.gen_value = self.gen_value;
    }

    /// Replaces: e200_print
    ///
    /// Renders the GenValue as the pass's `-debug-only=set-send-dst-re` dump does, delegating the
    /// value itself to the simple or composite form.
    pub(crate) fn print(&self, out: &mut String) {
        match self.gen_value {
            GenValue::Unknown => out.push_str("(GenValue: Unknown)"),
            GenValue::Simple(simple) => {
                out.push_str("kSimple:");
                simple.print(out);
            }
            GenValue::Composite(composite) => {
                out.push_str("kComposite:");
                composite.print(out);
            }
        }
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
    use super::{GenValue, SendDestination, SetDstGenValueLxlu};
    use crate::islands::sentient::dialects::{Op, Val, sentient};
    use crate::transform::sentient::set_send_destination_re::QueryMapOp;

    /// `sentient.set_send_dst`'s stand-in — whatever op the GenValue points at.
    fn nop() -> Op {
        Op::Sentient(sentient::Op::Nop { dbg_name: None })
    }

    /// A GenValue with both base flags set.
    fn flagged(value: SetDstGenValueLxlu) -> SetDstGenValueLxlu {
        SetDstGenValueLxlu {
            is_optimized: true,
            is_dead: true,
            ..value
        }
    }

    /// e198 — the kind's own value decides, the flags do not, and an unknown matches nothing.
    #[test]
    fn e198_compares_within_a_kind_and_never_across_or_from_unknown() {
        let pt = SetDstGenValueLxlu::simple(SendDestination::Pt, nop());
        assert!(
            flagged(pt.clone()).is_equal(&pt),
            "the flags must not count"
        );
        assert!(!pt.is_equal(&SetDstGenValueLxlu::simple(SendDestination::Sfp, nop())));

        let qmap = SetDstGenValueLxlu::composite(QueryMapOp::of(Val(7), Val(3), Val(4)), nop());
        assert!(qmap.is_equal(&qmap));
        assert!(!qmap.is_equal(&pt), "simple against composite is false");

        // ⛔ Unlike the implicit-sync twin, two unknowns are NOT equal.
        let unknown = SetDstGenValueLxlu::unknown();
        assert!(!unknown.is_equal(&unknown));
    }

    /// e199 — the value and `op_` travel; `is_optimized_` and `is_dead_` stay behind.
    #[test]
    fn e199_copies_the_value_and_op_but_not_the_flags() {
        let source = flagged(SetDstGenValueLxlu::simple(SendDestination::L0su, nop()));
        let mut target = SetDstGenValueLxlu::unknown();

        source.copy_to(&mut target);

        assert!(matches!(target.gen_value(), GenValue::Simple(_)));
        assert!(target.is_equal(&SetDstGenValueLxlu::simple(SendDestination::L0su, nop())));
        assert_eq!(target.op(), Some(&nop()));
        assert!(!target.is_optimized, "is_optimized_ is not copied");
        assert!(!target.is_dead, "is_dead_ is not copied");
    }

    /// e200 — each kind's prefix, the delegated value, and one suffix per flag.
    #[test]
    fn e200_prefixes_the_kind_and_appends_both_suffixes() {
        let mut unknown = String::new();
        flagged(SetDstGenValueLxlu::unknown()).print(&mut unknown);
        assert_eq!(unknown, "(GenValue: Unknown) - optimized! - dead!");

        let mut simple = String::new();
        SetDstGenValueLxlu::simple(SendDestination::Pt, nop()).print(&mut simple);
        assert_eq!(simple, "kSimple:(GenValue: PT)");

        let mut composite = String::new();
        SetDstGenValueLxlu::composite(QueryMapOp::of(Val(7), Val(3), Val(4)), nop())
            .print(&mut composite);
        assert_eq!(
            composite,
            "kComposite:%7 = uniform.query_map(map:%3, key:%4) : index"
        );
    }
}
