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

//! `ReadOnlyRegisterRenumbering.cpp` — 7 of the campaign's 656 units (dependency level(s) [0, 1, 2]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e127_runOn` | 127 | 0 | 6 | `dcc/src/Transform/Sentient/ReadOnlyRegisterRenumbering.cpp:102` |
//! | `e128_doRenumbering` | 128 | 0 | 15 | `dcc/src/Transform/Sentient/ReadOnlyRegisterRenumbering.cpp:144` |
//! | `e129_cleanup` | 129 | 0 | 1 | `dcc/src/Transform/Sentient/ReadOnlyRegisterRenumbering.cpp:163` |
//! | `e130_runOn` | 130 | 0 | 46 | `dcc/src/Transform/Sentient/ReadOnlyRegisterRenumbering.cpp:199` |
//! | `e342_runOnOperation` | 342 | 1 | 6 | `dcc/src/Transform/Sentient/ReadOnlyRegisterRenumbering.cpp:109` |
//! | `e343_computeNewRegisterIndices` | 343 | 1 | 23 | `dcc/src/Transform/Sentient/ReadOnlyRegisterRenumbering.cpp:117` |
//! | `e456_runOn` | 456 | 2 | 15 | `dcc/src/Transform/Sentient/ReadOnlyRegisterRenumbering.cpp:183` |

// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET. `e342_runOnOperation` — the pass entry — has landed
// as [`ReadOnlyRegisterRenumbering::run_on_operation`], and nothing in this crate calls it, so every
// item below is still reachable only from this file's own tests. CI runs clippy with `-D warnings`.
// ⭐ REMOVE THIS WHEN A PIPELINE CALLS `run_on_operation`: from then on an unused item here is a real
// defect.
#![allow(dead_code)]

use crate::arch::Arch;
use crate::formats::Bits;
use crate::islands::sentient::dialects::{
    Definitions, Op, Val, sentient, set_value_reg_index, uniform,
};
use crate::islands::sentient::{Program, ProgramUnit};
use crate::model::Model;
use crate::transform::sentient::ProgStitch;
use crate::transform::sentient::analyses::{ExpressionEvaluator, PinningSchemeManager};
use crate::workload::Workload;

/// `-dcc-read-only-register-renumbering-disable`, `cl::init(false)` (`:58-61`).
const DISABLE_THIS_PASS: bool = false;

/// `-dcc-read-only-register-renumbering-force`, `cl::init(false)` (`:63-66`) — run the pass even when
/// the program is not being stitched.
const FORCE_IT_DESPITE_PROG_STITCH: bool = false;

/// THE LBR ADDRESS A CANDIDATE WAS INITIALISED WITH — `AddrTy val_`, the constant its copy reads.
///
/// ⛔ NON-NEGATIVE BY CONSTRUCTION. `using AddrTy = int64_t;  // must be signed` (`:68`) is signed so
/// that the `lbr_addr = -1` sentinel and the `DT_CHECK_MSG(lbr_addr >= 0)` that discharges it can be
/// written at all; what a [`Candidate`] stores has already passed that check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct LbrAddress(pub(crate) u64);

impl LbrAddress {
    /// The constant as an address — `None` is the reference's untouched `-1`.
    fn of(value: i64) -> Option<LbrAddress> {
        u64::try_from(value).ok().map(LbrAddress)
    }

    /// The address back as the `AddrTy` the evaluator is asked for.
    ///
    /// ⭐ EXACT FOR EVERY REACHABLE VALUE: [`LbrAddress::of`] is the only constructor and it keeps
    /// only what came from a non-negative `i64`, so the saturation is the crate's idiom for an arm
    /// nothing reaches rather than a conversion this loses information in.
    fn value(self) -> i64 {
        i64::try_from(self.0).unwrap_or(i64::MAX)
    }
}

/// ONE LBR COPY THIS PASS MAY RENUMBER — `Candidate` (`:70-91`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Candidate {
    /// `op_`, AS THE VALUE IT BINDS — ⛔ AN IDENTITY, NOT A BORROW, for the reason
    /// [`crate::transform::sentient::ForRef`] gives: the pass rewrites the body it collected from.
    /// `getOp()->getResult(0)` is what both readers use, and a `sentient.scalar_copy` binds one value,
    /// which is the `DT_CHECK_MSG(getNumResults() == 1)` of `doRenumbering` as a fact about the island.
    pub(crate) copy: Val,
    /// `val_`.
    pub(crate) address: LbrAddress,
    /// `element_size_` — ⭐ NEVER THE REFERENCE'S `-1`: that value returns before the candidate is
    /// built (`:236-241`).
    pub(crate) element_size: Bits,
    /// `new_register_index_` — ⛔ `None` IS THE REFERENCE'S **UNINITIALISED** FIELD. Its one mutator
    /// is called by `computeNewRegisterIndices` (e343) for every candidate, so reading it before that
    /// is undefined there and a named `todo!` here; see [`ReadOnlyRegisterRenumbering::do_renumbering`].
    pub(crate) new_register_index: Option<sentient::RegIndex>,
}

/// `ReadOnlyRegisterRenumberingPass`'s OWN STATE — the two fields these four units share
/// (`:173`, `:177`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ReadOnlyRegisterRenumbering {
    /// `unsafe_to_renumber_` — ⛔ ONCE SET IT IS NEVER CLEARED, and `cleanup()` does not touch it, so
    /// the first unit that gives up disables the pass for every unit after it.
    pub(crate) unsafe_to_renumber: bool,
    /// `candidates_`.
    pub(crate) candidates: Vec<Candidate>,
}

impl ReadOnlyRegisterRenumbering {
    /// Replaces: e342_runOnOperation
    ///
    /// The pass entry: two gates, then the module walk.
    ///
    /// ⛔ THE SECOND GATE IS THE PASS. Renumbering a read-only register is only ever asked for when
    /// the program is one piece of a stitched program, so on the crate's own standalone path
    /// ([`ProgStitch::Standalone`]) this pass does nothing at all — which is a fact about the
    /// pipeline, not a reason to drop the body.
    /// ⭐ `getOperation()` IS THE `ModuleOp`, and this island's [`Program`] IS that module, so
    /// `runOn(ModuleOp)` is [`Self::run_on_program`] (e127) rather than a third overload.
    pub(crate) fn run_on_operation<A: Arch, M: Model, W: Workload>(
        &mut self,
        program: &mut Program<A, M, W>,
        stitching: ProgStitch,
    ) {
        if DISABLE_THIS_PASS {
            return;
        }
        if matches!(stitching, ProgStitch::Standalone) && !FORCE_IT_DESPITE_PROG_STITCH {
            return;
        }
        self.run_on_program(program);
    }

    /// Replaces: e127_runOn
    ///
    /// Runs the pass over every `dataflow.program_unit` of the module.
    ///
    /// ⛔ NAMED FOR ITS ARGUMENT because `runOn(ModuleOp)`, `runOn(dataflow::ProgramUnitOp)` (e456)
    /// and `runOn(Operation *)` (e130) are one overload set in C++ and cannot all be `run_on` here.
    ///
    /// ⭐ `WalkResult::skip()` COSTS NOTHING HERE: a program unit cannot nest inside another in this
    /// island, so there is nothing for the walk to decline to descend into.
    pub(crate) fn run_on_program<A: Arch, M: Model, W: Workload>(
        &mut self,
        program: &mut Program<A, M, W>,
    ) {
        for unit in program.units.iter_mut() {
            self.run_on_unit(unit);
        }
    }

    /// `ReadOnlyRegisterRenumberingPass::runOn(dataflow::ProgramUnitOp)` — e127's ONE callee, and
    /// SENPASS UNIT e456, whose anchor is still open below.
    ///
    /// ⛔ THE SCHEDULER RECORDED THE EDGE INVERTED: `UNITS.tsv` gives e456 `calls e127_runOn` and e127
    /// no callees, because the two `runOn`s share a name. e127 is the caller. Isolating the delegation
    /// in a private seam is the `2a8195231` precedent, and e456's anchor is left untouched.
    fn run_on_unit<A: Arch>(&mut self, unit: &mut ProgramUnit<A>) {
        todo!(
            "ReadOnlyRegisterRenumberingPass::runOn(dataflow::ProgramUnitOp) — senpass e456 \
             (ReadOnlyRegisterRenumbering.cpp:183) is not ported yet, and this {} op unit needs it",
            unit.body.len()
        )
    }

    /// Replaces: e343_computeNewRegisterIndices
    ///
    /// Asks the pinning scheme where each candidate's constant address ended up, and records that
    /// position as the candidate's new register index.
    ///
    /// ⛔ THE PASS ORDER IS THE INVARIANT AND THE REFERENCE ABORTS ON IT: an empty map, a `-1` entry
    /// or two units disagreeing are three `DT_CHECK`s (`:130-143`), so they are `panic!`s here, never a
    /// candidate quietly skipped — a skipped one would reach [`Self::do_renumbering`] with no index.
    /// ⭐ WHICH ENTRY IS READ DOES NOT MATTER: `begin()->second` off a `DenseMap` has no defined order,
    /// and the third check is that every unit agreed.
    /// ⛔ `calls e132_setNewRegisterIndex` IN THE SCHEDULE IS A NAME COLLISION: e132 is
    /// `RegisterPacking`'s mutator on its own `Register`, not this file's one-line `Candidate` setter.
    pub(crate) fn compute_new_register_indices(
        &mut self,
        sps_manager: &dyn PinningSchemeManager,
        evaluator: &mut dyn ExpressionEvaluator,
    ) {
        if self.unsafe_to_renumber {
            return;
        }
        for candidate in &mut self.candidates {
            let val_ev = evaluator.constant(candidate.address.value());
            let unit_to_index =
                sps_manager.find_matching_pinned_addr(val_ev, candidate.element_size);
            let Some((_unit, first)) = unit_to_index.first() else {
                panic!("DT_CHECK(!unit_to_index.empty()) (`:130`) for {:?}", candidate.copy)
            };
            let Some(index) = *first else {
                panic!(
                    "DT_CHECK_MSG(index >= 0, \"Unable to find matching pinned address! Make sure to \
                     run this pass after the address-pinning pass.\") (`:131-135`)"
                )
            };
            if unit_to_index.iter().any(|(_unit, other)| *other != Some(index)) {
                panic!(
                    "DT_CHECK(.. \"Expecting same position index for all matching pinned \
                     addresses\") (`:136-143`): {unit_to_index:?}"
                )
            }
            candidate.new_register_index = Some(index);
        }
    }

    /// Replaces: e128_doRenumbering
    ///
    /// Writes each candidate's computed index onto its `sentient.scalar_copy`.
    ///
    /// ⛔ THE WHOLE EFFECT, and it is the island's `set_value_reg_index`: of `setValueRegIndex`'s long
    /// dispatch the only arm a candidate can reach is
    /// `copyOp.setRegIndexAttr(builder.getI32IntegerAttr(index))` (`SentientOps.cpp:2014-2018`).
    pub(crate) fn do_renumbering(&self, body: &mut [Op]) {
        if self.unsafe_to_renumber {
            return;
        }
        for candidate in &self.candidates {
            let Some(index) = candidate.new_register_index else {
                todo!(
                    "doRenumbering: candidate {:?} has no new register index — \
                     StaticPinningSchemeManager::findMatchingPinnedAddr, which \
                     computeNewRegisterIndices (senpass e343) reads it from, is out of campaign scope",
                    candidate.copy
                )
            };
            set_value_reg_index(body, candidate.copy, Some(index));
        }
    }

    /// Replaces: e129_cleanup
    ///
    /// Drops the candidate list between program units.
    ///
    /// ⛔ IT DOES NOT RESET `unsafe_to_renumber_`; see the field.
    pub(crate) fn cleanup(&mut self) {
        self.candidates.clear();
    }

    /// Replaces: e130_runOn
    ///
    /// Collects one LBR `sentient.scalar_copy` of a constant as a renumbering candidate — or gives up
    /// on the whole pass.
    ///
    /// ⛔ TWO GIVE-UP ARMS, AND THEY ARE PERMANENT (see [`Self::cleanup`]): a `uniform.query_map` whose
    /// per-core constants disagree, and an absent `element_size`.
    pub(crate) fn run_on_op(&mut self, op: &Op, defs: Definitions<'_>) {
        if self.unsafe_to_renumber {
            return;
        }
        let Op::Sentient(sentient::Op::ScalarCopy {
            input,
            result,
            reg,
            element_size,
            ..
        }) = op
        else {
            return;
        };
        if reg.locale != sentient::RegType::Lbr {
            return;
        }
        if !is_sentient_constant(*input, defs) {
            todo!(
                "runOn: DT_CHECK_MSG(isConstant<sentient::ConstantOp>(copy_op.getInp()), \
                 \"LBR has to be initialized from a constant\") on {input:?} (:204-205)"
            )
        }
        let address = match defs.of(*input) {
            Some(Op::Sentient(sentient::Op::ScalarConstant { value, .. })) => {
                LbrAddress::of(*value)
            }
            Some(Op::Uniform(uniform::Op::QueryMap { map, .. })) => {
                let const_values = constant_target_values(*map, defs);
                // ⭐ `isConstant(query_op)` IS THE CHECK ABOVE RE-ASKED, so it is already true and the
                // `&&` reduces to the `all_of`. `first() == None` is then unreachable — it would mean
                // a non-constant target — and taking the give-up arm for it invents nothing.
                match const_values.first() {
                    Some(first) if const_values.iter().all(|value| value == first) => {
                        LbrAddress::of(*first)
                    }
                    _ => {
                        self.unsafe_to_renumber = true;
                        return;
                    }
                }
            }
            // `AddrTy lbr_addr = -1;` left untouched, which the next line refuses.
            _ => None,
        };
        let Some(address) = address else {
            todo!(
                "runOn: DT_CHECK_MSG(lbr_addr >= 0, \"unable to identify the lbr address\") \
                 on {input:?} (:230)"
            )
        };
        let Some(element_size) = *element_size else {
            self.unsafe_to_renumber = true;
            return;
        };
        self.candidates.push(Candidate {
            copy: *result,
            address,
            element_size,
            new_register_index: None,
        });
    }
}

/// `dcc::utils::isConstant<sentient::ConstantOp>` (`Utils/Utils.cpp:423`) at this pass's one
/// instantiation — a `sentient.scalar_constant`, or a `uniform.query_map` all of whose per-core values
/// are one. ⛔ FALSE FOR A REGION ARGUMENT, which has no defining op.
///
/// ⭐ AN EMPTY MAPPING IS VACUOUSLY TRUE HERE, where the reference's
/// `DT_CHECK(!immutable_map.getValues().empty())` (`Utils/Utils.cpp:434`) aborts — and it still
/// refuses, one step later: [`constant_target_values`] names that same check on the empty mapping.
fn is_sentient_constant(val: Val, defs: Definitions<'_>) -> bool {
    let is_constant_op =
        |op: Option<&Op>| matches!(op, Some(Op::Sentient(sentient::Op::ScalarConstant { .. })));
    match defs.of(val) {
        Some(Op::Uniform(uniform::Op::QueryMap { map, .. })) => match defs.of(*map) {
            Some(Op::Uniform(uniform::Op::DefImmutableMapping { pairs, .. })) => pairs
                .iter()
                .all(|(_key, value)| is_constant_op(defs.of(*value))),
            _ => false,
        },
        op => is_constant_op(op),
    }
}

/// `dcc::uniform::utils::getConstantTargetValues` (`Dialect/Uniform/Utils.cpp:382`) — every target
/// value of a query map's mapping, as a number.
///
/// ⛔ NOT AN ANCHORED UNIT: `dcc/src/Dialect/` is outside this campaign's file list. ⭐ AN EMPTY ANSWER
/// IS THE REFERENCE'S OWN `const_values.clear()`: not all targets are `sentient.scalar_constant`.
fn constant_target_values(map: Val, defs: Definitions<'_>) -> Vec<i64> {
    let pairs = match defs.of(map) {
        Some(Op::Uniform(uniform::Op::DefImmutableMapping { pairs, .. })) => pairs.clone(),
        other => todo!(
            "getConstantTargetValues: a uniform.query_map's $map is not a \
             uniform.def_immutable_mapping (Dialect/Uniform/Utils.cpp:385): {other:?}"
        ),
    };
    if pairs.is_empty() {
        todo!(
            "getConstantTargetValues: DT_CHECK(!immutable_map.getValues().empty()) \
             (Dialect/Uniform/Utils.cpp:387)"
        )
    }
    let mut values = Vec::new();
    for (_key, value) in pairs {
        match defs.of(value) {
            Some(Op::Sentient(sentient::Op::ScalarConstant { value, .. })) => values.push(*value),
            _ => return Vec::new(),
        }
    }
    values
}

// crustify:todo: e456_runOn
//   authority : dcc/src/Transform/Sentient/ReadOnlyRegisterRenumbering.cpp:183  (15 body lines, level 2)
//   original  : void ReadOnlyRegisterRenumberingPass::runOn(dataflow::ProgramUnitOp unit)
//   calls     : e127_runOn, e128_doRenumbering, e129_cleanup, e130_runOn, e221_initialize, e343_computeNewRegisterIndices

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Dd2;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::dataflow_ir::{GroupId, OpIndex, ProgramName, Units};
    use crate::islands::sentient::ProgramUnits;
    use crate::islands::sentient::dialects::sentient::{Reg, RegIndex, RegType};
    use crate::transform::sentient::analyses::{
        Evaluation, EvaluatedValue, OffsetSites, OutOfScopeEvaluator, OutOfScopePinningSchemeManager,
    };
    use crate::units::DfirUnit;

    /// A model, so a program is typed; nothing here reads it.
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

    /// `%r = sentient.scalar_constant {value = N : si64} : index`.
    fn scalar_const(result: u32, value: i64) -> Op {
        Op::Sentient(sentient::Op::ScalarConstant {
            value,
            result: Val(result),
            reg_locale: RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    /// `%r = sentient.scalar_copy %in {element_size = .., regLocale = locale}`.
    fn copy(input: u32, result: u32, locale: RegType, element_size: Option<Bits>) -> Op {
        Op::Sentient(sentient::Op::ScalarCopy {
            input: Val(input),
            result: Val(result),
            reg: Reg {
                locale,
                index: None,
            },
            program_header: false,
            element_size,
        })
    }

    /// e130 — the vendor's own shape: an LBR copy of a `sentient.scalar_constant` carrying an
    /// `element_size` becomes exactly one candidate, with the constant as its address.
    #[test]
    fn run_on_op_collects_an_lbr_copy_of_a_constant() {
        let body = vec![
            scalar_const(1, 384),
            copy(1, 2, RegType::Lbr, Some(Bits(16))),
        ];
        let regions: [&[Op]; 1] = [&body];
        let defs = Definitions::from_innermost(&regions);
        let mut pass = ReadOnlyRegisterRenumbering::default();
        pass.run_on_op(&body[1], defs);
        assert_eq!(
            pass,
            ReadOnlyRegisterRenumbering {
                unsafe_to_renumber: false,
                candidates: vec![Candidate {
                    copy: Val(2),
                    address: LbrAddress(384),
                    element_size: Bits(16),
                    new_register_index: None,
                }],
            }
        );
    }

    /// e130's negative — no `element_size` attribute is the reference's `-1`, which gives up on the
    /// whole pass and collects nothing.
    #[test]
    fn run_on_op_gives_up_when_the_element_size_is_absent() {
        let body = vec![scalar_const(1, 384), copy(1, 2, RegType::Lbr, None)];
        let regions: [&[Op]; 1] = [&body];
        let defs = Definitions::from_innermost(&regions);
        let mut pass = ReadOnlyRegisterRenumbering::default();
        pass.run_on_op(&body[1], defs);
        assert_eq!(
            pass,
            ReadOnlyRegisterRenumbering {
                unsafe_to_renumber: true,
                candidates: Vec::new(),
            }
        );
    }

    /// e128 — the effect: the candidate's index lands on the copy's `regIndex`, and nothing else moves.
    #[test]
    fn do_renumbering_writes_the_new_index_onto_the_copy() {
        let mut body = vec![
            scalar_const(1, 384),
            copy(1, 2, RegType::Lbr, Some(Bits(16))),
        ];
        let pass = ReadOnlyRegisterRenumbering {
            unsafe_to_renumber: false,
            candidates: vec![Candidate {
                copy: Val(2),
                address: LbrAddress(384),
                element_size: Bits(16),
                new_register_index: Some(RegIndex::at::<7>()),
            }],
        };
        pass.do_renumbering(&mut body);
        assert_eq!(
            body,
            vec![
                scalar_const(1, 384),
                Op::Sentient(sentient::Op::ScalarCopy {
                    input: Val(1),
                    result: Val(2),
                    reg: Reg {
                        locale: RegType::Lbr,
                        index: Some(RegIndex::at::<7>()),
                    },
                    program_header: false,
                    element_size: Some(Bits(16)),
                }),
            ]
        );
    }

    /// e129 — the candidates go and the give-up flag STAYS, which is what poisons the next unit.
    #[test]
    fn cleanup_clears_the_candidates_and_leaves_the_give_up_flag() {
        let mut pass = ReadOnlyRegisterRenumbering {
            unsafe_to_renumber: true,
            candidates: vec![Candidate {
                copy: Val(2),
                address: LbrAddress(384),
                element_size: Bits(16),
                new_register_index: None,
            }],
        };
        pass.cleanup();
        assert_eq!(
            pass,
            ReadOnlyRegisterRenumbering {
                unsafe_to_renumber: true,
                candidates: Vec::new(),
            }
        );
    }

    /// A module holding one `dataflow.program_unit`, which is all any walk here needs.
    fn one_unit_program() -> Program<Dd2, AnyModel, AnyRung> {
        Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            preamble: Vec::new(),
            units: ProgramUnits::of(
                ProgramUnit {
                    on: Units::one(DfirUnit::L3lu, Val(0)),
                    precision: None,
                    body: vec![scalar_const(1, 384)],
                    arch: core::marker::PhantomData,
                },
                Vec::new(),
            ),
            bound: core::marker::PhantomData,
        }
    }

    /// e127 — every program unit is visited, and the visit is e456's, which is not ported.
    #[test]
    #[should_panic(expected = "senpass e456")]
    fn run_on_program_delegates_each_unit_to_the_unported_e456() {
        let mut program = one_unit_program();
        ReadOnlyRegisterRenumbering::default().run_on_program(&mut program);
    }

    /// e342 — the standalone compilation: the prog-stitch gate returns before the walk, so the module
    /// is never entered and the unported e456 is never reached.
    #[test]
    fn e342_run_on_operation_does_nothing_when_the_program_is_not_stitched() {
        let mut program = one_unit_program();
        let untouched = program.clone();
        ReadOnlyRegisterRenumbering::default()
            .run_on_operation(&mut program, ProgStitch::Standalone);
        assert_eq!(program, untouched);
    }

    /// e342's positive — stitching passes both gates and the pass walks the module, which is e127.
    #[test]
    #[should_panic(expected = "senpass e456")]
    fn e342_run_on_operation_walks_the_module_when_the_program_is_stitched() {
        let mut program = one_unit_program();
        ReadOnlyRegisterRenumbering::default().run_on_operation(&mut program, ProgStitch::Stitched);
    }

    /// e343 — the candidate's address is evaluated, the scheme is asked where it was pinned, and that
    /// position becomes the new register index. ⭐ TWO UNITS AGREEING is the reference's own check.
    #[test]
    fn e343_records_the_pinned_position_as_the_new_register_index() {
        struct StatedEvaluator;

        impl ExpressionEvaluator for StatedEvaluator {
            fn constant(&mut self, value: i64) -> EvaluatedValue {
                assert_eq!(value, 384);
                EvaluatedValue(11)
            }

            fn evaluate_value(&mut self, _value: Val) -> Evaluation {
                todo!("e343 evaluates no IR value, only its candidate's constant")
            }

            fn evaluate_sum(&mut self, _lhs: &Evaluation, _rhs: &Evaluation) -> Evaluation {
                todo!("e343 evaluates no sum")
            }

            fn build_offset_value(
                &mut self,
                _evaluation: &Evaluation,
                _sites: &mut OffsetSites<'_>,
                _walked: &mut Vec<Op>,
                _ty: ScalarTy,
            ) -> Val {
                todo!("e343 builds no value")
            }
        }

        struct StatedScheme;

        impl PinningSchemeManager for StatedScheme {
            fn find_matching_pinned_addr(
                &self,
                ev: EvaluatedValue,
                element_size: Bits,
            ) -> Vec<(Val, Option<RegIndex>)> {
                assert_eq!((ev, element_size), (EvaluatedValue(11), Bits(16)));
                vec![
                    (Val(90), Some(RegIndex::at::<5>())),
                    (Val(91), Some(RegIndex::at::<5>())),
                ]
            }
        }

        let mut pass = ReadOnlyRegisterRenumbering {
            unsafe_to_renumber: false,
            candidates: vec![Candidate {
                copy: Val(2),
                address: LbrAddress(384),
                element_size: Bits(16),
                new_register_index: None,
            }],
        };
        pass.compute_new_register_indices(&StatedScheme, &mut StatedEvaluator);
        assert_eq!(
            pass.candidates[0].new_register_index,
            Some(RegIndex::at::<5>())
        );
    }

    /// e343's negative — a poisoned pass asks NOTHING, which the two out-of-scope seams prove: either
    /// call would be a `todo!`.
    #[test]
    fn e343_asks_nothing_once_the_pass_has_given_up() {
        let mut pass = ReadOnlyRegisterRenumbering {
            unsafe_to_renumber: true,
            candidates: vec![Candidate {
                copy: Val(2),
                address: LbrAddress(384),
                element_size: Bits(16),
                new_register_index: None,
            }],
        };
        pass.compute_new_register_indices(
            &OutOfScopePinningSchemeManager,
            &mut OutOfScopeEvaluator,
        );
        assert_eq!(pass.candidates[0].new_register_index, None);
    }
}
