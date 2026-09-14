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

//! `MultiDimLoopPeeling.cpp` — 5 of the campaign's 656 units (dependency level(s) [0, 1, 3, 7]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e093_stringifyPeelingType` | 093 | 0 | 12 | `dcc/src/Transform/Sentient/MultiDimLoopPeeling.cpp:93` |
//! | `e094_insertOrUpdatePeelingType` | 094 | 0 | 20 | `dcc/src/Transform/Sentient/MultiDimLoopPeeling.cpp:164` |
//! | `e321_printLoopToPeelingType` | 321 | 1 | 11 | `dcc/src/Transform/Sentient/MultiDimLoopPeeling.cpp:188` |
//! | `e515_getOrCreateTupleForIV` | 515 | 3 | 8 | `dcc/src/Transform/Sentient/MultiDimLoopPeeling.cpp:149` |
//! | `e643_runOnOperation` | 643 | 7 | 34 | `dcc/src/Transform/Sentient/MultiDimLoopPeeling.cpp:738` |

pub(crate) mod loop_peeling_manager;

use std::collections::BTreeMap;

use crate::arch::Arch;
use crate::islands::dataflow_ir::Values;
use crate::islands::sentient::Program;
use crate::islands::sentient::dialects::{Definitions, Op, Val, sentient};
use crate::islands::sentient::print;
use crate::model::Model;
use crate::transform::sentient::ForRef;
use crate::transform::sentient::analyses::{
    ExpressionEvaluator, InstructionEstimator, PropagationAnalysis, UnitIndexMap,
};
use crate::transform::sentient::loop_tree::LoopTree;
use crate::transform::sentient::utils::{
    ForLoopInfo, NormalizedIv, SenTarget, for_loop_info_if_iv,
};
use crate::units::DfirUnit;
use crate::workload::Workload;

/// WHICH ITERATION(S) OF A LOOP GET PEELED — `LoopPeelingManager::PeelingType`
/// (`dcc/src/Transform/Sentient/MultiDimLoopPeeling.cpp:86-91`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeelingType {
    /// `kFirstIterOnly`.
    FirstIterOnly,
    /// `kLastIterOnly`.
    LastIterOnly,
    /// `kFirstAndLastIter`.
    FirstAndLastIter,
    /// `kNoPeeling`.
    NoPeeling,
}

impl PeelingType {
    /// Replaces: e093_stringifyPeelingType
    ///
    /// The spelling of one peeling mode.
    ///
    /// ⛔ TRAP: NOT DEBUG OUTPUT. `performLoopPeeling` feeds it to
    /// `updateDbgName("MDLP(", for_op, stringifyPeelingType(p_type) + ")")` (`:577-578`), so these
    /// four strings reach the emitted IR as a loop's `dbgName`.
    ///
    /// ⛔ TRAP: THE `default:` ARM IS `"NoPeeling"` (`:101-102`) — not an error, not the empty string.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::FirstIterOnly => "FirstIterOnly",
            Self::LastIterOnly => "LastIterOnly",
            Self::FirstAndLastIter => "FirstAndLastIter",
            Self::NoPeeling => "NoPeeling",
        }
    }
}

/// THE PEELING MODES THE CANDIDATE LIST MAY HOLD — [`PeelingType`] MINUS `kNoPeeling`.
///
/// ⛔⛔ THE TWO `DT_CHECK_MSG`s OF `insertOrUpdatePeelingType` ARE THIS TYPE. *"Map should only
/// contain valid peeling modes"* guards the argument (`:165-166`) and *"Expect valid peeling mode in
/// map"* guards the value already stored (`:176-177`); a list that cannot hold `kNoPeeling` makes
/// both unwritable — which is what the field's own comment asks for, *"No loop should have the
/// peeling type kNoPeeling in this list"* (`:128`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Peeling {
    /// `kFirstIterOnly`.
    FirstIterOnly,
    /// `kLastIterOnly`.
    LastIterOnly,
    /// `kFirstAndLastIter`.
    FirstAndLastIter,
}

impl Peeling {
    /// The mode `ty` names — `None` for `kNoPeeling`, which is an ABSENCE of peeling and not a
    /// refusal to peel.
    #[must_use]
    pub const fn of(ty: PeelingType) -> Option<Peeling> {
        match ty {
            PeelingType::FirstIterOnly => Some(Peeling::FirstIterOnly),
            PeelingType::LastIterOnly => Some(Peeling::LastIterOnly),
            PeelingType::FirstAndLastIter => Some(Peeling::FirstAndLastIter),
            PeelingType::NoPeeling => None,
        }
    }

    /// This mode as a [`PeelingType`] — for [`PeelingType::spelling`].
    #[must_use]
    pub const fn ty(self) -> PeelingType {
        match self {
            Peeling::FirstIterOnly => PeelingType::FirstIterOnly,
            Peeling::LastIterOnly => PeelingType::LastIterOnly,
            Peeling::FirstAndLastIter => PeelingType::FirstAndLastIter,
        }
    }
}

/// `LoopPeelingManager::loop_to_peeling_type_` — the loops to peel and how, in the order
/// `findCandidates` found them (`:127-130`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PeelingCandidates(Vec<(ForRef, Peeling)>);

impl PeelingCandidates {
    /// The records in insertion order — `findCandidates` answers `!empty()` (`:430`) and
    /// `performLoopPeeling` walks them in REVERSE, innermost loop out (`:547`).
    #[must_use]
    pub fn records(&self) -> &[(ForRef, Peeling)] {
        &self.0
    }

    /// Replaces: e094_insertOrUpdatePeelingType
    ///
    /// Records `peeling` for `for_op`, or unions it with the mode already recorded there.
    ///
    /// ⛔ TRAP: THE UNION IS "DIFFERS, SO BOTH", NOT A JOIN OVER THE THREE MODES. The reference
    /// writes `kFirstAndLastIter` whenever the stored mode merely DIFFERS from the new one
    /// (`:180-181`), so `kFirstAndLastIter` meeting `kFirstIterOnly` stays `kFirstAndLastIter` only
    /// because that rule happens to give the right answer there.
    pub fn insert_or_update(&mut self, for_op: ForRef, peeling: Peeling) {
        match self.0.iter_mut().find(|(loop_ref, _)| *loop_ref == for_op) {
            Some((_, existing)) => {
                if *existing != peeling {
                    *existing = Peeling::FirstAndLastIter;
                }
            }
            None => self.0.push((for_op, peeling)),
        }
    }

    /// Replaces: e321_printLoopToPeelingType
    ///
    /// The candidate list as the pass's `LLVM_DEBUG` reads it (`:656`): a mode line then the whole
    /// loop, per record — or the one line that says there is nothing to peel.
    ///
    /// ⛔ TRAP: `OS << record.first` PRINTS THE WHOLE `sentient.for`, body and all: the map key is
    /// the op, not a name for it, so one record costs as many lines as the loop has ops.
    /// ⭐ [`print::emit`] ENDS ITS OWN LINE, so the reference's trailing `<< "\n"` is already there
    /// and adding it again would blank-line every record.
    #[must_use]
    pub fn print_loop_to_peeling_type(&self, unit: &[Op]) -> String {
        let mut out = String::new();
        if self.0.is_empty() {
            out.push_str("No loops to be peeled.\n");
            return out;
        }
        out.push_str("Candidate loops for peeling:\n");
        for (loop_ref, peeling) in &self.0 {
            out.push_str(peeling.ty().spelling());
            out.push_str(":\n");
            if let Some(op) = for_op_at(unit, *loop_ref) {
                print::emit(&mut out, op, 0);
            }
        }
        out
    }
}

/// The `sentient.for` a [`ForRef`] names, wherever under `unit` it sits — `Operation::walk`.
fn for_op_at(unit: &[Op], loop_ref: ForRef) -> Option<&Op> {
    for op in unit {
        if let Op::Sentient(sentient::Op::For { iv, .. }) = op
            && *iv == loop_ref.0
        {
            return Some(op);
        }
        for region in crate::islands::sentient::dialects::regions_ref(op) {
            if let Some(found) = for_op_at(region, loop_ref) {
                return Some(found);
            }
        }
    }
    None
}

#[cfg(test)]
mod unit_tests {
    use super::{
        DfirUnit, IvLoopInfo, Model, Peeling, PeelingCandidates, PeelingType, Program, SenTarget,
        Values, Workload, run_on_operation,
    };
    use crate::arch::Dd2;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::{GroupId, OpIndex, ProgramName, Units};
    use crate::islands::sentient::dialects::{Definitions, Op, Val, sentient};
    use crate::islands::sentient::{ProgramUnit, ProgramUnits};
    use crate::transform::sentient::ForRef;
    use crate::transform::sentient::analyses::{
        OutOfScopeEvaluator, OutOfScopeInstructionEstimator, OutOfScopePropagationAnalysis,
        OutOfScopeUnitIndexMap,
    };

    /// The four spellings `updateDbgName` writes into `MDLP(..)`.
    #[test]
    fn stringify_peeling_type_spells_all_four_modes() {
        assert_eq!(PeelingType::FirstIterOnly.spelling(), "FirstIterOnly");
        assert_eq!(PeelingType::LastIterOnly.spelling(), "LastIterOnly");
        assert_eq!(PeelingType::FirstAndLastIter.spelling(), "FirstAndLastIter");
        assert_eq!(PeelingType::NoPeeling.spelling(), "NoPeeling");
        assert_eq!(Peeling::of(PeelingType::NoPeeling), None);
    }

    /// A second, DIFFERENT mode for a loop already recorded unions to `kFirstAndLastIter`; the same
    /// mode twice leaves it alone, and a second loop is a second record.
    #[test]
    fn insert_or_update_unions_only_a_differing_mode() {
        let (outer, inner) = (ForRef(Val(1)), ForRef(Val(2)));
        let mut candidates = PeelingCandidates::default();
        candidates.insert_or_update(outer, Peeling::FirstIterOnly);
        candidates.insert_or_update(outer, Peeling::FirstIterOnly);
        assert_eq!(candidates.records(), [(outer, Peeling::FirstIterOnly)]);
        candidates.insert_or_update(outer, Peeling::LastIterOnly);
        candidates.insert_or_update(inner, Peeling::LastIterOnly);
        assert_eq!(
            candidates.records(),
            [
                (outer, Peeling::FirstAndLastIter),
                (inner, Peeling::LastIterOnly)
            ]
        );
    }

    /// e515 — an induction variable resolves once and is then answered from the cache; anything
    /// else is not cached and stays `None`.
    #[test]
    fn get_or_create_tuple_for_iv_caches_only_a_hit() {
        let unit = vec![
            Op::Sentient(sentient::Op::ScalarConstant {
                value: 4,
                result: Val(9),
                reg_locale: sentient::RegType::Imm,
                ty: crate::islands::dataflow_ir::ty::ScalarTy::Index,
                is_symbol: false,
            }),
            Op::Sentient(sentient::Op::For {
                iv: Val(1),
                bound: Val(9),
                bound_reg: None,
                carried: Vec::new(),
                dbg_name: None,
                body: Vec::new(),
            }),
        ];
        let regions: [&[Op]; 1] = [&unit];
        let defs = Definitions::from_innermost(&regions);

        let mut cache = IvLoopInfo::default();
        let info = cache.get_or_create(Val(1), defs).expect("Val(1) is the IV");
        assert_eq!(info.loop_op, ForRef(Val(1)));
        assert_eq!((info.lower_bound, info.step, info.iterations), (4, -1, 4));
        // The second ask is the cache's, and the constant is not an induction variable at all.
        assert_eq!(cache.get_or_create(Val(1), defs), Some(info));
        assert_eq!(cache.get_or_create(Val(9), defs), None);
    }

    /// e321 — nothing to peel is one line; one record spells its mode and then prints the loop it
    /// names, body included.
    #[test]
    fn e321_prints_the_mode_then_the_whole_loop() {
        let unit = vec![Op::Sentient(sentient::Op::For {
            iv: Val(1),
            bound: Val(0),
            bound_reg: None,
            carried: Vec::new(),
            dbg_name: None,
            body: vec![Op::Sentient(sentient::Op::Nop { dbg_name: None })],
        })];
        let mut candidates = PeelingCandidates::default();
        assert_eq!(
            candidates.print_loop_to_peeling_type(&unit),
            "No loops to be peeled.\n"
        );
        candidates.insert_or_update(ForRef(Val(1)), Peeling::LastIterOnly);
        let printed = candidates.print_loop_to_peeling_type(&unit);
        assert!(printed.starts_with("Candidate loops for peeling:\nLastIterOnly:\n"));
        assert!(printed.contains("sentient.for "));
        assert!(printed.contains("sentient.nop"));
    }

    /// A model and a rung, so the program is typed; nothing this pass does reads either.
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

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct AnyRung;
    impl Workload for AnyRung {
        const ROWS: u32 = 1;
        const ACTIVE_CAP: u32 = 64;
    }

    /// One program unit holding one top-level `sentient.for`, on `on`.
    fn one_loop_on(on: DfirUnit) -> Program<Dd2, AnyModel, AnyRung> {
        Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            preamble: Vec::new(),
            units: ProgramUnits::of(
                ProgramUnit {
                    on: Units::one(on, Val(0)),
                    precision: None,
                    body: vec![Op::Sentient(sentient::Op::For {
                        iv: Val(1),
                        bound: Val(2),
                        bound_reg: None,
                        carried: Vec::new(),
                        dbg_name: None,
                        body: vec![Op::Sentient(sentient::Op::Nop { dbg_name: None })],
                    })],
                    arch: core::marker::PhantomData,
                },
                Vec::new(),
            ),
            bound: core::marker::PhantomData,
        }
    }

    /// e643 — an SFP unit's top-level loop reaches `loop_peeling_manager::run`, whose first act is
    /// to ask the under-estimating estimator how much IBUFF is left.
    #[test]
    #[should_panic(expected = "getRemainingIbuffSpace")]
    fn e643_peels_the_top_level_loops_of_an_sfp_unit() {
        let mut program = one_loop_on(DfirUnit::Sfp);
        run_on_operation(
            &mut program,
            &mut OutOfScopeInstructionEstimator,
            &mut OutOfScopeEvaluator,
            &mut OutOfScopePropagationAnalysis,
            &OutOfScopeUnitIndexMap,
            &mut Values::default(),
            SenTarget::Sentient,
        );
    }

    /// e643's negative control: `is_any_of(unit_comp, SFP, PE, PT)` skips an LX unit outright, so the
    /// estimator is never asked and the loop is left standing.
    #[test]
    fn e643_skips_a_unit_that_is_not_sfp_pe_or_pt() {
        let mut program = one_loop_on(DfirUnit::Lxlu);
        let before = program.units.iter().next().expect("one unit").body.clone();

        run_on_operation(
            &mut program,
            &mut OutOfScopeInstructionEstimator,
            &mut OutOfScopeEvaluator,
            &mut OutOfScopePropagationAnalysis,
            &OutOfScopeUnitIndexMap,
            &mut Values::default(),
            SenTarget::Sentient,
        );

        assert_eq!(program.units.iter().next().expect("one unit").body, before);
    }
}

/// `LoopPeelingManager::iv_to_loop_info_` — every value this pass has already resolved as an
/// induction variable (`MultiDimLoopPeeling.cpp:132`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IvLoopInfo(BTreeMap<Val, ForLoopInfo>);

impl IvLoopInfo {
    /// Replaces: e515_getOrCreateTupleForIV
    ///
    /// The loop and bounds `val` names as a BARE induction variable, resolved once and remembered.
    ///
    /// ⛔ TRAP: ONLY A HIT IS CACHED (`:155`), so a `val` that is not an induction variable is
    /// re-resolved on every ask — the reference's `{nullptr,-1,-1,-1,-1}` is the `None` here.
    /// ⛔ TRAP: `allow_normalized_iv` IS `false` (`:153`) — a `const - iv` is NOT this loop's IV.
    pub fn get_or_create(&mut self, val: Val, defs: Definitions<'_>) -> Option<ForLoopInfo> {
        if let Some(cached) = self.0.get(&val) {
            return Some(*cached);
        }
        let info = for_loop_info_if_iv(val, NormalizedIv::Rejected, defs)?;
        self.0.insert(val, info);
        Some(info)
    }
}

/// `DisableThisPass` — the `-dcc-multi-dim-loop-peeling-disable` `cl::opt`, `cl::init(false)`
/// (`:36-39`).
const DISABLE_THIS_PASS: bool = false;

/// Replaces: e643_runOnOperation
///
/// THE PASS ENTRY (`:738-771`): peels every TOP-LEVEL loop of every SFP, PE or PT program unit.
///
/// ⛔ ONE [`IvLoopInfo`] PER LOOP — `iv_to_loop_info_` is a per-manager field (`:132`) and the
/// reference builds one `LoopPeelingManager` per top-level loop.
/// ⛔ THE SIBLING CHAIN IS TAKEN BEFORE ANY PEELING (`:759-760`), so the loops peeling creates are not
/// themselves visited; `markAnalysesPreserved` (`:770`) is pass-manager bookkeeping and is dropped.
pub fn run_on_operation<A: Arch, M: Model, W: Workload>(
    program: &mut Program<A, M, W>,
    under_estimating_ie: &mut impl InstructionEstimator,
    evaluator: &mut impl ExpressionEvaluator,
    propagation: &mut impl PropagationAnalysis,
    unit_index_map: &impl UnitIndexMap,
    values: &mut Values,
    sen_target: SenTarget,
) {
    if DISABLE_THIS_PASS {
        return;
    }
    let Program {
        preamble, units, ..
    } = program;
    for unit in units.iter_mut() {
        // `DT_CHECK_MSG(get_unit_op, "Cannot determine GetUnitOp!")` (`:744`) HAS NO ARM HERE: a
        // [`ProgramUnit`] always names the units it runs on.
        if !matches!(
            unit.on.kind(),
            DfirUnit::Sfp | DfirUnit::Pe | DfirUnit::PtRow(_)
        ) {
            continue;
        }
        let tree: LoopTree<false> = LoopTree::of(&unit.body);
        if tree.empty() {
            continue;
        }
        let mut top_level = Vec::new();
        let mut node = tree.first_child(tree.root());
        while let Some(n) = node {
            // `dyn_cast_or_null<sentient::ForOp>(n->getOperation())` — a node the tree kept but whose
            // op is not a `sentient.for` is skipped, not peeled.
            if let Some(for_op) = tree.loop_of(n) {
                top_level.push(for_op);
            }
            node = tree.next_sibling(n);
        }
        for outer_loop in top_level {
            let mut ivs = IvLoopInfo::default();
            loop_peeling_manager::run(
                outer_loop,
                unit,
                preamble,
                &mut ivs,
                under_estimating_ie,
                evaluator,
                propagation,
                unit_index_map,
                values,
                sen_target,
            );
        }
    }
}
