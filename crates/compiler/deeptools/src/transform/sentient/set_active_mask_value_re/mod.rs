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

//! `SetActiveMaskValueRE.cpp` — 7 of the campaign's 656 units (dependency level(s) [0, 1, 2, 3, 4]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e182_copyTo` | 182 | 0 | 8 | `dcc/src/Transform/Sentient/SetActiveMaskValueRE.hpp:38` |
//! | `e183_print` | 183 | 0 | 6 | `dcc/src/Transform/Sentient/SetActiveMaskValueRE.hpp:55` |
//! | `e184_maskValuesAreEquivalent` | 184 | 0 | 6 | `dcc/src/Transform/Sentient/SetActiveMaskValueRE.hpp:63` |
//! | `e374_isEqual` | 374 | 1 | 8 | `dcc/src/Transform/Sentient/SetActiveMaskValueRE.hpp:30` |
//! | `e473_runOn` | 473 | 2 | 18 | `dcc/src/Transform/Sentient/SetActiveMaskValueRE.cpp:54` |
//! | `e535_runOn` | 535 | 3 | 4 | `dcc/src/Transform/Sentient/SetActiveMaskValueRE.cpp:73` |
//! | `e582_runOnOperation` | 582 | 4 | 5 | `dcc/src/Transform/Sentient/SetActiveMaskValueRE.cpp:78` |

// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET, so `SetActiveMaskValueGenValue` is reachable only
// from `set_active_mask_value_rde_tree` and its tests until `e582_runOnOperation` (level 4) lands. CI
// runs clippy with `-D warnings`, so without this the first ported leaf of the module fails the gate.
// ⭐ REMOVE THIS WITH e582: at that point an unused item here is a real defect again.
#![allow(dead_code)]

use crate::arch::Arch;
use crate::islands::sentient::dialects::sentient::{RawPrecision, SliceId, ValidEntries, WslLen};
use crate::islands::sentient::dialects::{Op, Val, sentient};
use crate::islands::sentient::print;
use crate::islands::sentient::{Program, ProgramUnit};
use crate::model::Model;
use crate::workload::Workload;

pub(crate) mod set_active_mask_value_rde_tree;

/// A `sentient.samv`'S INHERENT ATTRIBUTE DICTIONARY — `samv_op->getAttrDictionary()`, which is what
/// `SetActiveMaskValueGenValue::attrs_` holds and compares.
///
/// ⛔ THE REFERENCE'S DICTIONARY ALSO CARRIES THE DISCARDABLE ATTRIBUTES `regIndices`/`regLocales`
/// (`SentientOps.td:1015-1016`), which the island's `Op::Samv` does not model — so two `samv`s
/// differing only in an assigned register compare EQUAL here and unequal in the reference.
///
/// ⛔ THE DICTIONARY IS THE OP'S ATTRIBUTES AND `$mask_value` IS AN OPERAND (`SentientOps.td:1019`),
/// so the mask value is NOT in here — it is the GenValue's other field, compared by its own rule.
///
/// ⛔ `dbgName` IS IN THE DICTIONARY, hence in the equality: `OptionalAttr<StrAttr>:$dbgName`
/// (`SentientOps.td:1026`) is an attribute like the other six, so two otherwise identical `samv`s with
/// different debug names are NOT equal definitions. That is the reference's behaviour, not a choice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SamvAttrs {
    /// `maskall`.
    pub(crate) mask_all: bool,
    /// `numvalidentry`.
    pub(crate) num_valid_entry: ValidEntries,
    /// `sliceid_xsl`.
    pub(crate) slice_id_xsl: SliceId,
    /// `xslinner`.
    pub(crate) xsl_inner: bool,
    /// `wsllen`.
    pub(crate) wsl_len: WslLen,
    /// `precision` — ⛔ a raw ISA field, never the `Precision` enum's spelling.
    pub(crate) precision: RawPrecision,
    /// `dbgName`.
    pub(crate) dbg_name: Option<String>,
}

impl SamvAttrs {
    /// `OS << attrs_` — MLIR's `DictionaryAttr` rendering, which sorts by name.
    ///
    /// ⛔ ALPHABETICAL, NOT DECLARATION ORDER, and an absent `dbgName` is absent from the dictionary
    /// rather than printed empty — the same rule the island's own `samv` printer follows.
    fn print(&self, out: &mut String) {
        let mut entries = vec![
            format!("maskall = {}", self.mask_all),
            format!("numvalidentry = {} : i32", self.num_valid_entry.0),
            format!("precision = {} : i32", self.precision.0),
            format!("sliceid_xsl = {} : i32", self.slice_id_xsl.0),
            format!("wsllen = {} : i32", self.wsl_len.0),
            format!("xslinner = {}", self.xsl_inner),
        ];
        if let Some(name) = &self.dbg_name {
            entries.push(format!("dbgName = \"{name}\""));
        }
        entries.sort();
        out.push('{');
        out.push_str(&entries.join(", "));
        out.push('}');
    }
}

/// `SetActiveMaskValueGenValue` (`SetActiveMaskValueRE.hpp:22`) — the dataflow definition an RDE node
/// generates for the SAMV state.
///
/// ⭐ DECLARED HERE, WHERE ITS OWN METHODS BELONG: e182-e184 are `copyTo`/`print`/
/// `maskValuesAreEquivalent` on this class and e374 is its `isEqual`, all scheduled into this file.
/// [`set_active_mask_value_rde_tree`]'s e180 constructs it — a batch filling the remaining anchors
/// should UNION with this, not duplicate it.
#[derive(Debug, Clone, Default)]
pub(crate) struct SetActiveMaskValueGenValue {
    /// `mask_value_` — `None` is the constructor's `nullptr`.
    mask_value: Option<Val>,
    /// `attrs_` — `None` is the constructor's null `DictionaryAttr`.
    attrs: Option<SamvAttrs>,
    /// `op_` — absent for the default-constructed unknown value.
    op: Option<Op>,
    /// `DataFlowDefinitionBase::is_optimized_`
    /// (`Analyses/RedundantDefinitionEliminationTree.hpp:327`) — the base class is OUT OF CAMPAIGN
    /// SCOPE, but e183 prints this flag, so the subclass holds it exactly as it holds `op_`.
    is_optimized: bool,
    /// `DataFlowDefinitionBase::is_dead_`, printed by e183 for the same reason.
    is_dead: bool,
}

impl SetActiveMaskValueGenValue {
    /// `SetActiveMaskValueGenValue()` — the unknown value.
    #[must_use]
    pub(crate) fn unknown() -> SetActiveMaskValueGenValue {
        SetActiveMaskValueGenValue::default()
    }

    /// `SetActiveMaskValueGenValue(mask_value, attrs, op)`.
    #[must_use]
    pub(crate) fn of(mask_value: Val, attrs: SamvAttrs, op: Op) -> SetActiveMaskValueGenValue {
        SetActiveMaskValueGenValue {
            mask_value: Some(mask_value),
            attrs: Some(attrs),
            op: Some(op),
            is_optimized: false,
            is_dead: false,
        }
    }

    /// `SetActiveMaskValueGenValue(samv_op.getMaskValue(), samv_op->getAttrDictionary(), op)` — the
    /// definition a `sentient.samv` generates, and nothing for any other operation.
    ///
    /// ⭐ ONE MATCH, so the mask value and the dictionary cannot be taken from two different ops.
    #[must_use]
    pub(crate) fn of_samv(op: &Op) -> Option<SetActiveMaskValueGenValue> {
        let Op::Sentient(sentient::Op::Samv {
            mask_value,
            mask_all,
            num_valid_entry,
            slice_id_xsl,
            xsl_inner,
            wsl_len,
            precision,
            dbg_name,
        }) = op
        else {
            return None;
        };
        Some(SetActiveMaskValueGenValue::of(
            *mask_value,
            SamvAttrs {
                mask_all: *mask_all,
                num_valid_entry: *num_valid_entry,
                slice_id_xsl: *slice_id_xsl,
                xsl_inner: *xsl_inner,
                wsl_len: *wsl_len,
                precision: *precision,
                dbg_name: dbg_name.clone(),
            },
            op.clone(),
        ))
    }

    /// `isUnknownValue()` (`SetActiveMaskValueRE.hpp:47`) — `!mask_value_ || !attrs_`.
    #[must_use]
    pub(crate) const fn is_unknown_value(&self) -> bool {
        self.mask_value.is_none() || self.attrs.is_none()
    }

    /// `getMaskValue()`.
    #[must_use]
    pub(crate) const fn mask_value(&self) -> Option<Val> {
        self.mask_value
    }

    /// `setMaskValue(v)`.
    pub(crate) const fn set_mask_value(&mut self, v: Option<Val>) {
        self.mask_value = v;
    }

    /// `getAttrs()`.
    #[must_use]
    pub(crate) const fn attrs(&self) -> Option<&SamvAttrs> {
        self.attrs.as_ref()
    }

    /// `setAttrs(a)`.
    pub(crate) fn set_attrs(&mut self, a: Option<SamvAttrs>) {
        self.attrs = a;
    }

    /// The op that generated it.
    #[must_use]
    pub(crate) const fn op(&self) -> Option<&Op> {
        self.op.as_ref()
    }
}

impl SetActiveMaskValueGenValue {
    /// Replaces: e182_copyTo
    ///
    /// Copies the mask value, the attribute dictionary and the generating op onto `lhs`, leaving its
    /// two base flags alone.
    ///
    /// ⛔ "COPY EVERYTHING EXCEPT THE `is_optimized_` FLAG" — and except `is_dead_`, which the
    /// reference's three assignments also leave untouched. `*lhs = self.clone()` would clobber both.
    ///
    /// ⛔ THE `dynamic_cast` AND ITS `DT_CHECK_MSG` BECAME THE PARAMETER TYPE: a definition that is
    /// not a `SetActiveMaskValueGenValue` is not expressible at this call, so nothing is checked at
    /// run time.
    pub(crate) fn copy_to(&self, lhs: &mut SetActiveMaskValueGenValue) {
        lhs.set_mask_value(self.mask_value());
        lhs.set_attrs(self.attrs.clone());
        lhs.op = self.op.clone();
    }

    /// Replaces: e183_print
    ///
    /// Renders the GenValue as the pass's `-debug-only` dump does.
    ///
    /// ⛔ THE REFERENCE PRINTS `mask_value_.getAsOpaquePointer()`, A MACHINE ADDRESS — nondeterministic
    /// between runs, so the port prints the value's SSA identity (`%14`) instead. A null `Value`
    /// streams as `0x0` there and as `%null` here; a null `attrs_` streams as MLIR's
    /// `<<NULL ATTRIBUTE>>`, which is kept verbatim.
    ///
    /// ⛔ THE `>)` IS THE REFERENCE'S OWN TYPO — an unmatched `>` with no opening `<` anywhere in the
    /// format. Preserved, because this dump is compared against the reference's.
    pub(crate) fn print(&self, out: &mut String) {
        out.push_str("(GenValue: value_:");
        match self.mask_value {
            Some(mask_value) => out.push_str(&print::val(mask_value)),
            None => out.push_str("%null"),
        }
        out.push_str(", attrs:");
        match &self.attrs {
            Some(attrs) => attrs.print(out),
            None => out.push_str("<<NULL ATTRIBUTE>>"),
        }
        out.push_str(">)");
        if self.is_optimized {
            out.push_str(" - optimized!");
        }
        if self.is_dead {
            out.push_str(" - dead!");
        }
    }

    /// Replaces: e184_maskValuesAreEquivalent
    ///
    /// Two mask values are equivalent when they are the same SSA value — deliberately shallow.
    ///
    /// ⛔ NOT `SetMaskGenValue`'S RULE. That sibling also folds two `sentient.constant`s with equal
    /// values (e189); this one does not, and the reference says why: the specialized canonicalization
    /// pass already commoned them, so a deeper comparison is future work.
    ///
    /// ⭐ AN ASSOCIATED FUNCTION: the reference's `const` member reads no field, and `Option` carries
    /// the nullability its `Value` parameters have, so two null masks are equivalent.
    #[must_use]
    pub(crate) fn mask_values_are_equivalent(a: Option<Val>, b: Option<Val>) -> bool {
        a == b
    }
}

// crustify:todo: e374_isEqual
//   authority : dcc/src/Transform/Sentient/SetActiveMaskValueRE.hpp:30  (8 body lines, level 1)
//   original  : bool isEqual(const DataFlowDefinitionBase &rhs) const final override
//   calls     : e184_maskValuesAreEquivalent

/// `Statistic<"samv_re_count", "num-samv-eliminated", "Number of times SAMV operations removed or
/// hoisted">` (`Transform/Sentient/Passes.td:57`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct SamvReCount(pub(crate) u32);

/// Replaces: e473_runOn
///
/// Builds this unit's SAMV redundant-definition tree, simplifies it, and banks how many SAMVs the
/// optimizer removed or hoisted.
///
/// ⛔ EVERY LINE OF THE BODY IS BLOCKED, AND BY TWO DIFFERENT THINGS: the tree's construction,
/// `compute` and `simplify` are `RedundantDefinitionEliminationTree`'s and OUT OF CAMPAIGN SCOPE,
/// while `optimize()` is this campaign's own e378. The two `print`s in between are debug output.
/// ⛔ THERE IS NO UNIT-KIND GATE HERE, unlike [`super::set_mask_re::run_on_unit`]'s `!= PT`: SAMV
/// redundancy elimination runs on every program unit (`:54-70`).
pub(crate) fn run_on_unit<A: Arch>(
    unit: &mut ProgramUnit<A>,
    samv_re_count: &mut SamvReCount,
) -> ! {
    let _ = (unit, samv_re_count);
    todo!(
        "e473_runOn: SetActiveMaskValueRDETree's construction, compute() and simplify() \
         (Analyses/RedundantDefinitionEliminationTree.hpp) are out of campaign scope, and \
         RDETreeOptimizer::optimize() is not ported yet (senpass e378, SetMaskRE.cpp:311) — together \
         they are the whole of SetActiveMaskValueRE.cpp:54-70"
    )
}

/// Replaces: e535_runOn
///
/// Runs SAMV redundant-definition elimination over every program unit of one module.
///
/// ⛔ NAMED FOR ITS ARGUMENT: `runOn(ModuleOp)` and `runOn(dataflow::ProgramUnitOp)` (e473) are one
/// C++ overload set and cannot both be `run_on` here.
/// ⛔ THE STATISTIC IS ASSIGNED PER UNIT, NOT ACCUMULATED (`:70`) — after the walk it holds the LAST
/// unit's count, not the module's total. That is the reference's behaviour.
/// ⭐ THE PREORDER WALK IS DROPPABLE MECHANISM: a program's units are a flat list here.
pub(crate) fn run_on_program<A: Arch, M: Model, W: Workload>(
    program: &mut Program<A, M, W>,
    samv_re_count: &mut SamvReCount,
) {
    for unit in program.units.iter_mut() {
        run_on_unit(unit, samv_re_count);
    }
}

// crustify:todo: e582_runOnOperation
//   authority : dcc/src/Transform/Sentient/SetActiveMaskValueRE.cpp:78  (5 body lines, level 4)
//   original  : void runOnOperation()
//   calls     : e473_runOn, e535_runOn

#[cfg(test)]
mod unit_tests {
    use super::{
        SamvAttrs, SamvReCount, SetActiveMaskValueGenValue, run_on_program, run_on_unit,
    };
    use crate::arch::Dd2;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::{GroupId, OpIndex, ProgramName, Units};
    use crate::islands::sentient::dialects::sentient::{
        RawPrecision, SliceId, ValidEntries, WslLen,
    };
    use crate::islands::sentient::dialects::{Op, Val, sentient};
    use crate::islands::sentient::{Program, ProgramUnit, ProgramUnits};
    use crate::model::Model;
    use crate::units::DfirUnit;
    use crate::workload::Workload;

    /// A model, so the program is typed; nothing here reads it.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct AnyModel;
    impl Model for AnyModel {
        const QUERY_HEADS: u32 = 32;
        const KV_HEADS: u32 = 8;
        const HEAD_DIM: u32 = 64;
        const HIDDEN: u32 = 2048;
        const LAYERS: u32 = 40;
        const FFN: u32 = 8192;
        const VOCAB: u32 = 49152;
    }

    /// A decode rung, for the same reason.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct AnyRung;
    impl Workload for AnyRung {
        const ROWS: u32 = 1;
        const ACTIVE_CAP: u32 = 64;
    }

    /// One empty program unit on `kind`.
    fn unit_on(kind: DfirUnit) -> ProgramUnit<Dd2> {
        ProgramUnit {
            on: Units::one(kind, Val(0)),
            precision: None,
            body: Vec::new(),
            arch: core::marker::PhantomData,
        }
    }

    /// One program with one unit, holding nothing — the walk reads only the unit list.
    fn program_on(kind: DfirUnit) -> Program<Dd2, AnyModel, AnyRung> {
        Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            preamble: Vec::new(),
            units: ProgramUnits::of(unit_on(kind), Vec::new()),
            bound: core::marker::PhantomData,
        }
    }

    /// `sentient.samv mask_value(%7) {...}` — the op a GenValue is built from.
    fn samv(mask_value: Val) -> Op {
        Op::Sentient(sentient::Op::Samv {
            mask_value,
            mask_all: false,
            num_valid_entry: ValidEntries(5),
            slice_id_xsl: SliceId(4),
            xsl_inner: false,
            wsl_len: WslLen(2),
            precision: RawPrecision(16),
            dbg_name: None,
        })
    }

    /// The attribute dictionary [`samv`] carries.
    fn attrs() -> SamvAttrs {
        SetActiveMaskValueGenValue::of_samv(&samv(Val(7)))
            .and_then(|value| value.attrs().cloned())
            .expect("a samv carries its dictionary")
    }

    /// A GenValue with both base flags set.
    fn flagged(value: SetActiveMaskValueGenValue) -> SetActiveMaskValueGenValue {
        SetActiveMaskValueGenValue {
            is_optimized: true,
            is_dead: true,
            ..value
        }
    }

    /// e182 — the mask value, the dictionary and `op_` travel; the two flags stay behind.
    #[test]
    fn copy_to_moves_the_value_the_attrs_and_the_op_but_not_the_flags() {
        let source = flagged(SetActiveMaskValueGenValue::of(
            Val(7),
            attrs(),
            samv(Val(7)),
        ));
        let mut target = SetActiveMaskValueGenValue::unknown();

        source.copy_to(&mut target);

        assert_eq!(target.mask_value(), Some(Val(7)));
        assert_eq!(target.attrs(), Some(&attrs()));
        assert_eq!(target.op(), Some(&samv(Val(7))));
        assert!(!target.is_optimized, "is_optimized_ is not copied");
        assert!(!target.is_dead, "is_dead_ is not copied");
        assert!(!target.is_unknown_value(), "both halves arrived");
    }

    /// e183 — the dictionary is alphabetical, the null halves keep their spellings, and each flag adds
    /// its own suffix.
    #[test]
    fn print_renders_the_dictionary_alphabetically_and_both_suffixes() {
        let mut out = String::new();
        SetActiveMaskValueGenValue::of(Val(7), attrs(), samv(Val(7))).print(&mut out);
        assert_eq!(
            out,
            "(GenValue: value_:%7, attrs:{maskall = false, numvalidentry = 5 : i32, \
             precision = 16 : i32, sliceid_xsl = 4 : i32, wsllen = 2 : i32, xslinner = false}>)"
        );

        let mut unknown = String::new();
        flagged(SetActiveMaskValueGenValue::unknown()).print(&mut unknown);
        assert_eq!(
            unknown,
            "(GenValue: value_:%null, attrs:<<NULL ATTRIBUTE>>>) - optimized! - dead!"
        );
    }

    /// e184 — the same SSA value and nothing else, two nulls included.
    #[test]
    fn only_the_same_ssa_value_is_an_equivalent_mask() {
        assert!(SetActiveMaskValueGenValue::mask_values_are_equivalent(
            Some(Val(7)),
            Some(Val(7))
        ));
        assert!(!SetActiveMaskValueGenValue::mask_values_are_equivalent(
            Some(Val(7)),
            Some(Val(8))
        ));
        assert!(!SetActiveMaskValueGenValue::mask_values_are_equivalent(
            Some(Val(7)),
            None
        ));
        assert!(SetActiveMaskValueGenValue::mask_values_are_equivalent(
            None, None
        ));
    }

    /// e473 — every unit is visited, and the tree it builds is out of campaign scope down to the
    /// optimizer, which is e378 and not ported.
    #[test]
    #[should_panic(expected = "senpass e378")]
    fn any_unit_reaches_the_unported_rde_tree_optimizer() {
        run_on_unit(&mut unit_on(DfirUnit::Lxlu), &mut SamvReCount(0));
    }

    /// e535 — there is no unit-kind gate: the first unit of the module reaches the per-unit body,
    /// which is e473 and blocked on the out-of-scope tree down to the unported e378.
    #[test]
    #[should_panic(expected = "senpass e378")]
    fn every_unit_of_the_module_is_run_on() {
        run_on_program(&mut program_on(DfirUnit::Pe), &mut SamvReCount(0));
    }
}
