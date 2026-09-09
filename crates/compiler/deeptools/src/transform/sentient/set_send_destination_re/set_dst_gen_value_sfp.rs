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
//! | `e203_isEqual` | 203 | 0 | 35 | `dcc/src/Transform/Sentient/SetSendDestinationRE.cpp:418` |
//! | `e204_copyTo` | 204 | 0 | 23 | `dcc/src/Transform/Sentient/SetSendDestinationRE.cpp:454` |
//! | `e205_print` | 205 | 0 | 19 | `dcc/src/Transform/Sentient/SetSendDestinationRE.cpp:478` |

use super::QueryMapOp;
use super::composite_set_dst_gen_value_sfp::CompositeSetDstGenValueSfp;
use super::simple_set_dst_gen_value_sfp::SimpleSetDstGenValueSfp;
use crate::islands::sentient::dialects::{
    Definitions, Op, uniform_mapping_keys, uniform_mapping_values,
};
use crate::units::{Core, Corelet};

/// `SetDstGenValueSFP::Kind` AND ITS `union GenValue` AS ONE VALUE
/// (`SetSendDestinationRE.hpp:175-179,208-217`) — see
/// [`super::set_dst_gen_value_lxlu::GenValue`] for why the tag and the union are one thing here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum GenValue {
    /// `kUnknown` — the most pessimistic mode, and the default-constructed value.
    #[default]
    Unknown,
    /// `kSimple` — the destination is a `dataflow.get_unit`, i.e. one core and corelet.
    Simple(SimpleSetDstGenValueSfp),
    /// `kComposite` — the destination comes out of a `uniform.query_map`.
    Composite(CompositeSetDstGenValueSfp),
}

/// `SetDstGenValueSFP` (`SetSendDestinationRE.hpp:173`) — the dataflow definition an RDE node
/// generates when this pass is optimizing the SFP's `SETDEST`.
///
/// ⭐ THE BASE CLASS'S THREE FIELDS ARE HELD HERE, for the reason
/// [`super::set_dst_gen_value_lxlu::SetDstGenValueLxlu`] records.
#[derive(Debug, Clone, Default)]
pub(crate) struct SetDstGenValueSfp {
    gen_value: GenValue,
    /// `op_` — absent for the default-constructed unknown value.
    op: Option<Op>,
    /// `is_optimized_`; nothing in scope sets it yet.
    is_optimized: bool,
    /// `is_dead_`.
    is_dead: bool,
}

impl SetDstGenValueSfp {
    /// `SetDstGenValueSFP()` — the unknown value.
    #[must_use]
    pub(crate) fn unknown() -> SetDstGenValueSfp {
        SetDstGenValueSfp::default()
    }

    /// `SetDstGenValueSFP(core_id, corelet_id, op)` (`:188`).
    #[must_use]
    pub(crate) fn simple(core: Core, corelet: Corelet, op: Op) -> SetDstGenValueSfp {
        SetDstGenValueSfp {
            gen_value: GenValue::Simple(SimpleSetDstGenValueSfp::of(core, corelet)),
            op: Some(op),
            is_optimized: false,
            is_dead: false,
        }
    }

    /// `SetDstGenValueSFP(qmap, op)` (`:183`).
    #[must_use]
    pub(crate) fn composite(qmap: QueryMapOp, op: Op) -> SetDstGenValueSfp {
        SetDstGenValueSfp {
            gen_value: GenValue::Composite(CompositeSetDstGenValueSfp::of(qmap)),
            op: Some(op),
            is_optimized: false,
            is_dead: false,
        }
    }

    /// `isUnknownValue()` — `!isSimple() && !isComposite()` (`:202-204`).
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

    /// Replaces: e203_isEqual
    ///
    /// Two SFP GenValues are equal when both are known, of the same kind, and agree on that kind's
    /// value — the destination core AND corelet, or the query map.
    ///
    /// ⭐⭐ THE COMPOSITE CASE HAS A SECOND CHANCE THE LXLU TWIN DOES NOT: two DIFFERENT
    /// `uniform.query_map` ops are still equal when they resolve to the same key list and the same
    /// value list (`SetSendDestinationRE.cpp:440-449`), which is why `defs` is a parameter here.
    /// ⛔ THE ORDER OF BOTH LISTS COUNTS — the reference's own todo says so; positional equality is
    /// the contract, not set equality.
    ///
    /// ⛔ AN UNKNOWN IS EQUAL TO NOTHING, `is_optimized`/`is_dead`/`op_` are left out, and the
    /// `dynamic_cast`'s `DT_CHECK_MSG` became the parameter type — as for e198.
    #[must_use]
    pub(crate) fn is_equal(&self, rhs: &SetDstGenValueSfp, defs: Definitions<'_>) -> bool {
        match (self.gen_value, rhs.gen_value) {
            (GenValue::Unknown, _) | (_, GenValue::Unknown) => false,
            (GenValue::Simple(lhs), GenValue::Simple(rhs)) => {
                lhs.core() == rhs.core() && lhs.corelet() == rhs.corelet()
            }
            (GenValue::Composite(lhs), GenValue::Composite(rhs)) => {
                let (lhs, rhs) = (lhs.query_map(), rhs.query_map());
                lhs == rhs
                    || (uniform_mapping_keys(lhs.key(), defs)
                        == uniform_mapping_keys(rhs.key(), defs)
                        && uniform_mapping_values(lhs.map(), lhs.key(), defs)
                            == uniform_mapping_values(rhs.map(), rhs.key(), defs))
            }
            (GenValue::Simple(_), GenValue::Composite(_))
            | (GenValue::Composite(_), GenValue::Simple(_)) => false,
        }
    }

    /// Replaces: e204_copyTo
    ///
    /// Copies the generating operation and the generated value — both ids, for a simple one — onto
    /// `lhs`, leaving its two flags alone.
    ///
    /// ⛔ THE FLAGS DO NOT TRAVEL, and `DT_CHECK_MSG(lhs_p->kind_ == Kind::kUnknown, "mixed mode
    /// copying ...")` is discharged by the tagged [`GenValue`]: see e199, which this mirrors exactly.
    pub(crate) fn copy_to(&self, lhs: &mut SetDstGenValueSfp) {
        lhs.op = self.op.clone();
        lhs.gen_value = self.gen_value;
    }

    /// Replaces: e205_print
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
    use super::{GenValue, SetDstGenValueSfp, SimpleSetDstGenValueSfp};
    use crate::islands::dataflow_ir::dialects::uniform;
    use crate::islands::sentient::dialects::{Definitions, Op, Val, dataflow, sentient};
    use crate::transform::sentient::set_send_destination_re::QueryMapOp;
    use crate::units::{Core, Corelet};

    /// `sentient.set_send_dst`'s stand-in — whatever op the GenValue points at.
    fn nop() -> Op {
        Op::Sentient(sentient::Op::Nop { dbg_name: None })
    }

    /// Core 0's corelet 0, and core 0's corelet 1.
    fn ids(corelet: u32) -> (Core, Corelet) {
        (
            Core::checked(0).expect("core 0"),
            Corelet::checked(corelet).expect("a corelet of this arch"),
        )
    }

    /// A GenValue with both base flags set.
    fn flagged(value: SetDstGenValueSfp) -> SetDstGenValueSfp {
        SetDstGenValueSfp {
            is_optimized: true,
            is_dead: true,
            ..value
        }
    }

    /// A scope binding `%0`/`%1` as `sfp` units and `%2`/`%3` as two mappings that answer alike.
    fn scope() -> Vec<Op> {
        let unit = |result| {
            Op::Dataflow(dataflow::Op::GetUnit {
                result,
                residency: crate::units::Residency::Global,
                unit: crate::units::DfirUnit::Sfp,
                num_folds: None,
            })
        };
        vec![
            unit(Val(0)),
            unit(Val(1)),
            Op::Uniform(uniform::Op::DefImmutableMapping {
                result: Val(2),
                pairs: vec![(Val(0), Val(1))],
            }),
            Op::Uniform(uniform::Op::DefImmutableMapping {
                result: Val(3),
                pairs: vec![(Val(0), Val(1))],
            }),
        ]
    }

    /// e203 — both ids decide a simple pair, and two DIFFERENT query maps that resolve alike are
    /// equal while an unknown matches nothing.
    #[test]
    fn e203_compares_both_ids_and_falls_back_to_the_resolved_key_and_value_lists() {
        let body = scope();
        let regions: [&[Op]; 1] = [&body];
        let defs = Definitions::from_innermost(&regions);

        let (core, corelet) = ids(0);
        let here = SetDstGenValueSfp::simple(core, corelet, nop());
        assert!(
            flagged(here.clone()).is_equal(&here, defs),
            "flags must not count"
        );
        let (other_core, other_corelet) = ids(1);
        assert!(!here.is_equal(
            &SetDstGenValueSfp::simple(other_core, other_corelet, nop()),
            defs
        ));

        // Distinct `query_map` ops over distinct mappings that hold the same pair: the identity test
        // fails and the key/value lists carry it.
        let via_2 = SetDstGenValueSfp::composite(QueryMapOp::of(Val(4), Val(2), Val(0)), nop());
        let via_3 = SetDstGenValueSfp::composite(QueryMapOp::of(Val(5), Val(3), Val(0)), nop());
        assert!(via_2.is_equal(&via_3, defs), "the resolved lists agree");
        assert!(
            !via_2.is_equal(&here, defs),
            "composite against simple is false"
        );

        let unknown = SetDstGenValueSfp::unknown();
        assert!(!unknown.is_equal(&unknown, defs));
    }

    /// e204 — both ids and `op_` travel; `is_optimized_` and `is_dead_` stay behind.
    #[test]
    fn e204_copies_both_ids_and_the_op_but_not_the_flags() {
        let (core, corelet) = ids(1);
        let source = flagged(SetDstGenValueSfp::simple(core, corelet, nop()));
        let mut target = SetDstGenValueSfp::unknown();

        source.copy_to(&mut target);

        assert_eq!(
            target.gen_value(),
            GenValue::Simple(SimpleSetDstGenValueSfp::of(core, corelet))
        );
        assert_eq!(target.op(), Some(&nop()));
        assert!(!target.is_optimized, "is_optimized_ is not copied");
        assert!(!target.is_dead, "is_dead_ is not copied");
    }

    /// e205 — each kind's prefix, the delegated value, and one suffix per flag.
    #[test]
    fn e205_prefixes_the_kind_and_appends_both_suffixes() {
        let mut unknown = String::new();
        flagged(SetDstGenValueSfp::unknown()).print(&mut unknown);
        assert_eq!(unknown, "(GenValue: Unknown) - optimized! - dead!");

        let (core, corelet) = ids(1);
        let mut simple = String::new();
        SetDstGenValueSfp::simple(core, corelet, nop()).print(&mut simple);
        assert_eq!(simple, "kSimple:(GenValue: core_id<0>, corelet_id<1>)");

        let mut composite = String::new();
        SetDstGenValueSfp::composite(QueryMapOp::of(Val(7), Val(3), Val(4)), nop())
            .print(&mut composite);
        assert_eq!(
            composite,
            "kComposite:%7 = uniform.query_map(map:%3, key:%4) : index"
        );
    }
}
