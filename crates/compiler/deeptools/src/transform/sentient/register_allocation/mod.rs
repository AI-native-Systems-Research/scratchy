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

//! `RegisterAllocation.cpp` — 2 of the campaign's 656 units (dependency level(s) [1, 2]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e344_naivelyAllocate` | 344 | 1 | 281 | `dcc/src/Transform/Sentient/RegisterAllocation.cpp:39` |
//! | `e457_runOnOperation` | 457 | 2 | 1 | `dcc/src/Transform/Sentient/RegisterAllocation.cpp:321` |


// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET — `e457_runOnOperation` is its one-line driver and is
// now filled, but it is the pass ENTRY, so filling it added no caller: everything here is still
// reachable only from this file's own tests. CI runs clippy with `-D warnings`. ⭐ REMOVE THIS WHEN THE
// SENTIENT PIPELINE CALLS `RegisterAllocation::run_on_operation`.
#![allow(dead_code)]

use crate::arch::Arch;
use crate::islands::sentient::dialects::sentient::{RegIndex, RegType, Reg};
use crate::islands::sentient::dialects::{self, Op, UniformRegions, Val, sentient, uniform};
use crate::islands::sentient::Program;
use crate::model::Model;
use crate::workload::Workload;

/// ONE REGISTER FILE'S NEXT FREE INDEX — one of the reference's `int` counters (`:40-48`).
///
/// ⛔ NOT AN `int`, AND [`NextIndex::release`] IS WHY: `lccr_counter--` (`:97`) can take the
/// reference's counter BELOW zero, and a negative index is this island's `None` — the exact `-1` the
/// backend refuses with `Register initialization out of boundary`. Saturating keeps the counter a
/// register index; the TRAP is stated at [`RegisterAllocation::naively_allocate`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct NextIndex(u32);

impl NextIndex {
    /// The next index WITHOUT advancing — the two xrf pointers, whose `++` is commented out.
    fn peek(self) -> RegIndex {
        RegIndex::allocated(self.0)
    }

    /// `reg_indices.push_back(counter); counter++;`
    fn take(&mut self) -> RegIndex {
        let index = self.peek();
        self.0 += 1;
        index
    }

    /// `lccr_counter--` (`:97`) — see the type's note for why it stops at zero.
    fn release(&mut self) {
        self.0 = self.0.saturating_sub(1);
    }
}

/// THE COUNTERS THE WALK CARRIES (`:40-48`).
///
/// ⛔ `ebr_counter` IS DECLARED AND NEVER READ (`:48`): no arm of the reference's walk hands out an
/// EBR index, so a ninth field here would be one nothing could ever move.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Counters {
    /// `lccr_counter` — ⭐ THE LOOP NESTING DEPTH, by the `for`/`yield` pair: a loop takes the next
    /// index on the way in and its `sentient.yield` gives it back on the way out.
    lccr: NextIndex,
    /// `jcr_counter`.
    jcr: NextIndex,
    /// `lrf_counter`.
    lrf: NextIndex,
    /// `xrf_rd_counter`.
    xrf_rd: NextIndex,
    /// `xrf_wr_counter`.
    xrf_wr: NextIndex,
    /// `lar_counter`.
    lar: NextIndex,
    /// `lbr_counter`.
    lbr: NextIndex,
    /// `ear_counter`.
    ear: NextIndex,
}

impl Counters {
    /// `isa<dataflow::ProgramUnitOp>(op)` (`:52-54`) — the five that restart per program unit.
    ///
    /// ⛔ `lar`/`lbr`/`ear` ARE NOT RESET, and that is the reference's own list: an address register is
    /// handed out across the whole module, a compute register per unit.
    fn enter_program_unit(&mut self) {
        self.lccr = NextIndex::default();
        self.jcr = NextIndex::default();
        self.lrf = NextIndex::default();
        self.xrf_rd = NextIndex::default();
        self.xrf_wr = NextIndex::default();
    }

    /// The next index in `locale`'s file, with the counter advanced where the reference advances it.
    ///
    /// ⛔ THE TWO XRF POINTERS DO NOT ADVANCE — `// xrf_rd_counter += 1; Already handled in the above
    /// passes.` (`:80-86`), so every xrf pointer in one unit is handed the SAME index.
    /// ⭐ `None` IS "NO COUNTER FOR THIS FILE", which [`RegisterAllocation::next_index`] reports as the
    /// op's own `emitError`.
    fn hand_out(&mut self, locale: RegType) -> Option<RegIndex> {
        match locale {
            RegType::Lrf => Some(self.lrf.take()),
            RegType::Jcr => Some(self.jcr.take()),
            RegType::Lccr => Some(self.lccr.take()),
            RegType::Lar => Some(self.lar.take()),
            RegType::Lbr => Some(self.lbr.take()),
            RegType::Ear => Some(self.ear.take()),
            RegType::XrfRdPtr => Some(self.xrf_rd.peek()),
            RegType::XrfWrPtr => Some(self.xrf_wr.peek()),
            // ⛔ NO `_` ARM: a register file this pass grows a counter for must be a build error here,
            // not an index silently taken from another file's.
            RegType::Unknown
            | RegType::Imm
            | RegType::Ebr
            | RegType::Gtr
            | RegType::Mvr
            | RegType::Unrelated => None,
        }
    }
}

/// WHICH REGISTER FILES ONE OP FORM ACCEPTS — the `if / else if` chain each arm of the walk spells out,
/// as the six distinct sets they are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Accepts {
    /// `sentient.for` (`:71-90`) — the only arm that takes `lccr`.
    Loop,
    /// `sentient.if` (`:104-121`) and `uniform.uniformize_regions` (`:135-153`).
    Branch,
    /// `sentient.scalar_add`, `scalar_sub`, `scalar_copy` (`:161-217`).
    Scalar,
    /// `sentient.load_and_send`, `receive_and_store` (`:220-259`).
    Transfer,
    /// `sentient.load_and_extract_scalar` (`:271-276`) and `load_compute_and_send` (`:305-310`).
    LrfOnly,
    /// `sentient.load_and_store` (`:294-300`).
    Address,
}

impl Accepts {
    /// Whether this op form's chain has an arm for `locale`.
    fn takes(self, locale: RegType) -> bool {
        match self {
            Accepts::Loop => matches!(
                locale,
                RegType::Lrf | RegType::Jcr | RegType::Lccr | RegType::XrfRdPtr | RegType::XrfWrPtr
            ),
            Accepts::Branch => {
                matches!(locale, RegType::Jcr | RegType::XrfRdPtr | RegType::XrfWrPtr)
            }
            Accepts::Scalar => matches!(
                locale,
                RegType::Lrf | RegType::Jcr | RegType::XrfRdPtr | RegType::XrfWrPtr
            ),
            Accepts::Transfer => matches!(locale, RegType::Lrf | RegType::Lar | RegType::Lbr),
            Accepts::LrfOnly => matches!(locale, RegType::Lrf),
            Accepts::Address => matches!(locale, RegType::Lar | RegType::Ear),
        }
    }
}

/// `op->emitError(..); signalPassFailure();` AS DATA — one op whose register locale this pass has no
/// arm for.
///
/// ⭐ NOT A `Result` AND NOT A PANIC, for the reason
/// [`crate::transform::sentient::enhanced_dead_variable_elimination`] gives: `signalPassFailure` is a
/// flag on the pass object, and the walk CONTINUES past the op it refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct UnknownLocale {
    /// The value that keeps no register — an identity, for the reason [`super::ForRef`] gives.
    pub(crate) at: Val,
    /// What it asked for, or `None` where the op carries no `regLocale` attribute at all.
    pub(crate) locale: Option<RegType>,
    /// The message the reference emitted.
    pub(crate) message: &'static str,
}

/// `"Unknown locale at the add operation"` — ⛔ THE `sentient.for` ARM'S MESSAGE TOO (`:88`), which is
/// the reference's own copy-paste and is kept verbatim.
const UNKNOWN_AT_ADD: &str = "Unknown locale at the add operation";

/// `RegisterAllocationPass`'s STATE — the counters, and every refusal it recorded.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct RegisterAllocation {
    counters: Counters,
    failures: Vec<UnknownLocale>,
}

impl RegisterAllocation {
    /// Replaces: e344_naivelyAllocate
    ///
    /// Walks the module in pre-order handing every register-carrying op the next index in its own
    /// register file, restarting the five per-unit counters at each `dataflow.program_unit`.
    ///
    /// ⛔ TRAP: `lccr_counter--` AT A `sentient.yield` MAKES LCCR THE LOOP NESTING DEPTH, and the
    /// reference would take it NEGATIVE for a loop whose entry 0 is not `lccr`
    /// (`RegisterTypeAssignment.cpp:443-447` emits no such loop); [`NextIndex::release`] stops at zero.
    /// ⛔ TRAP: A REFUSED ENTRY ABANDONS THE WHOLE OP — the reference builds `reg_indices` locally and
    /// `setRegIndicesAttr` is the last statement, so an unknown locale leaves EVERY entry of that op
    /// unassigned while the counters it already took stay taken.
    /// ⭐ THE `use_empty() -> push(-1)` ARM (`:62-69`) IS UNREACHABLE ON THIS ISLAND: it tests
    /// `i - getNumRegionIterArgs() - 1 >= 0`, the RESULT half of the reference's `1 + 2n` array, and
    /// [`sentient::Carried`] collapses each argument-and-result pair into one slot the loop always needs.
    pub(crate) fn naively_allocate<A: Arch, M: Model, W: Workload>(
        &mut self,
        program: &mut Program<A, M, W>,
    ) {
        // The module's own ops, which a `ModuleOp` pre-order walk reaches before any program unit.
        self.walk(&mut program.preamble, Parent::Other);
        for unit in program.units.iter_mut() {
            self.counters.enter_program_unit();
            self.walk(&mut unit.body, Parent::Other);
        }
    }

    /// Every `emitError` + `signalPassFailure` this pass made.
    #[must_use]
    pub(crate) fn failures(&self) -> &[UnknownLocale] {
        &self.failures
    }

    /// `module_op.walk<WalkOrder::PreOrder>` (`:50`) — the op, then its regions.
    fn walk(&mut self, scope: &mut [Op], parent: Parent) {
        for op in scope.iter_mut() {
            self.allocate_for(op, parent);
            // ⛔ WHOSE REGION THE NESTED SCOPE IS, because the `sentient.yield` arm asks exactly that.
            let inner = match op {
                Op::Sentient(sentient::Op::For { .. }) => Parent::For,
                _ => Parent::Other,
            };
            for region in dialects::regions_mut(op) {
                self.walk(region, inner);
            }
        }
    }

    /// The next index for `locale` where this op form has an arm for it, and the recorded
    /// `emitError` + `signalPassFailure` where it does not (`:87-89` and its eight siblings).
    fn next_index(
        &mut self,
        at: Val,
        locale: RegType,
        accepts: Accepts,
        message: &'static str,
    ) -> Option<RegIndex> {
        if accepts.takes(locale) {
            if let Some(index) = self.counters.hand_out(locale) {
                return Some(index);
            }
        }
        self.failures.push(UnknownLocale {
            at,
            locale: Some(locale),
            message,
        });
        None
    }

    /// One op of the walk — the `if / else if` chain over op kinds (`:51-315`).
    fn allocate_for(&mut self, op: &mut Op, parent: Parent) {
        match op {
            Op::Sentient(sentient::Op::For {
                iv,
                bound_reg,
                carried,
                ..
            }) => {
                // ⭐ ENTRY 0 FIRST, THEN ONE PER CARRIED VALUE — the array order of
                // [`sentient::Op::For::bound_reg`].
                let asked: Vec<(Val, RegType)> = bound_reg
                    .as_ref()
                    .map(|reg| (*iv, reg.locale))
                    .into_iter()
                    .chain(carried.iter().map(|slot| (slot.arg, slot.reg.locale)))
                    .collect();
                let Some(indices) = self.indices_for(&asked, Accepts::Loop, UNKNOWN_AT_ADD) else {
                    return;
                };
                let mut assigned = indices.into_iter();
                if let Some(reg) = bound_reg {
                    reg.index = assigned.next();
                }
                for slot in carried.iter_mut() {
                    slot.reg.index = assigned.next();
                }
            }
            Op::Sentient(sentient::Op::Yield { .. }) => {
                // `isa<sentient::ForOp>(yield_op->getParentRegion()->getParentOp())` (`:96-99`).
                if matches!(parent, Parent::For) {
                    self.counters.lccr.release();
                }
            }
            Op::Sentient(sentient::Op::If { yielded, .. }) => {
                let asked: Vec<(Val, RegType)> = yielded
                    .iter()
                    .map(|slot| (slot.result, slot.reg.locale))
                    .collect();
                let message = "Only JCR/XRFRDPTR/XRFWRPTR locales are accepted for if-ops";
                let Some(indices) = self.indices_for(&asked, Accepts::Branch, message) else {
                    return;
                };
                for (slot, index) in yielded.iter_mut().zip(indices) {
                    slot.reg.index = Some(index);
                }
            }
            // `else if (auto unif_regions = dyn_cast<uniform::UniformizeRegionsOp>(op))` (`:127-159`),
            // whose whole body is inside `if (unif_regions->hasAttr(regLocales))`.
            //
            // ⛔ NEITHER ISLAND SPELLING CARRIES THAT ATTRIBUTE, which is a gap recorded at
            // [`dialects::value_reg_locale`] and not one this unit can close: `uniform` at this rung IS
            // the DataflowIR dialect (`dialects::uniform`), so the sentient register arrays cannot go on
            // it, and the sentient-rung [`UniformRegions`] spelling has no printer for them either.
            // Every `uniform.uniformize_regions` here is therefore the reference's own no-attribute
            // case, which assigns nothing — and `LiveRangeReduction.cpp:1071`, whose
            // `setRegLocalesAttr` on a rebuilt uniformize op is the one writer in the tree, is the unit
            // that has to mint the field before this arm has anything to do.
            Op::Uniform(uniform::Op::UniformizeRegions { .. })
            | Op::UniformRegions(UniformRegions::UniformizeRegions { .. }) => {}
            Op::Sentient(sentient::Op::ScalarAdd { result, reg, .. }) => {
                self.allocate_singular(*result, reg.as_mut(), Accepts::Scalar, UNKNOWN_AT_ADD);
            }
            Op::Sentient(sentient::Op::ScalarSub { result, reg, .. }) => {
                let message = "Unknown locale at the sub operation";
                self.allocate_singular(*result, reg.as_mut(), Accepts::Scalar, message);
            }
            Op::Sentient(sentient::Op::ScalarCopy { result, reg, .. }) => {
                let message = "Unknown locale at the copy operation";
                self.allocate_singular(*result, Some(reg), Accepts::Scalar, message);
            }
            Op::Sentient(sentient::Op::LoadAndSend { result, reg, .. }) => {
                let message = "Unknown locale at the load_and_send operation";
                self.allocate_singular(*result, Some(reg), Accepts::Transfer, message);
            }
            Op::Sentient(sentient::Op::ReceiveAndStore { result, reg, .. }) => {
                let message = "Unknown locale at the receive_and_store operation";
                self.allocate_singular(*result, Some(reg), Accepts::Transfer, message);
            }
            // ⭐ THE TWO ARRAY OPS' `DT_CHECK_MSG(reg_locales.size() == kNumRegisters)` (`:266`, `:288`)
            // IS A FACT OF THIS ISLAND: each declares one named [`Reg`] per result, so the two counts
            // cannot disagree.
            Op::Sentient(sentient::Op::LoadAndExtractScalar {
                addr_result,
                data_result,
                addr_reg,
                data_reg,
                ..
            }) => {
                let asked = [
                    (*addr_result, addr_reg.locale),
                    (*data_result, data_reg.locale),
                ];
                let message = "Unknown locale at the load_and_extract_scalar operation";
                if let Some(indices) = self.indices_for(&asked, Accepts::LrfOnly, message) {
                    addr_reg.index = Some(indices[0]);
                    data_reg.index = Some(indices[1]);
                }
            }
            Op::Sentient(sentient::Op::LoadAndStore {
                results,
                src_reg,
                dst_reg,
                ..
            }) => {
                let asked = [(results.0, src_reg.locale), (results.1, dst_reg.locale)];
                let message = "Unknown locale at the load_and_store operation";
                if let Some(indices) = self.indices_for(&asked, Accepts::Address, message) {
                    src_reg.index = Some(indices[0]);
                    dst_reg.index = Some(indices[1]);
                }
            }
            Op::Sentient(sentient::Op::LoadComputeAndSend { result, reg, .. }) => {
                let message = "Unknown locale at the load_compute_and_send operation";
                self.allocate_singular(*result, Some(reg), Accepts::LrfOnly, message);
            }
            // ⭐ THE REFERENCE'S CHAIN HAS NO `else`: an op it names no arm for keeps whatever register
            // an earlier pass gave it, which is what leaves e457's `-1`s where nothing allocates.
            _ => {}
        }
    }

    /// The whole array, or nothing — see [`Self::naively_allocate`]'s second TRAP.
    fn indices_for(
        &mut self,
        asked: &[(Val, RegType)],
        accepts: Accepts,
        message: &'static str,
    ) -> Option<Vec<RegIndex>> {
        let mut indices = Vec::with_capacity(asked.len());
        for (at, locale) in asked {
            indices.push(self.next_index(*at, *locale, accepts, message)?);
        }
        Some(indices)
    }

    /// `op.setRegIndexAttr(builder.getI32IntegerAttr(counter))` — the ops carrying ONE `regLocale`.
    ///
    /// ⛔ `None` IS THE ABSENT ATTRIBUTE, which `getRegLocale()` cannot answer any arm of the chain
    /// with, so it lands on the same `emitError`.
    fn allocate_singular(
        &mut self,
        at: Val,
        reg: Option<&mut Reg>,
        accepts: Accepts,
        message: &'static str,
    ) {
        match reg {
            Some(reg) => {
                if let Some(index) = self.next_index(at, reg.locale, accepts, message) {
                    reg.index = Some(index);
                }
            }
            None => self.failures.push(UnknownLocale {
                at,
                locale: None,
                message,
            }),
        }
    }
}

/// WHOSE REGION THE WALK IS IN — the one question this pass asks about a parent (`:96-99`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Parent {
    /// A `sentient.for` body, where a `sentient.yield` gives the LCCR index back.
    For,
    /// Anything else, including the program unit body itself.
    Other,
}

impl RegisterAllocation {
    /// Replaces: e457_runOnOperation
    ///
    /// The pass entry: hand every register-carrying op in the module its index
    /// ([`Self::naively_allocate`]).
    ///
    /// ⛔ NO GATE AND NO OPTION — unlike its siblings this pass has neither a `DisableThisPass` flag
    /// nor a component test, so `runOnOperation` is the whole delegation.
    /// ⭐ `getOperation()` IS THE `ModuleOp`, which this island spells as the [`Program`] the walk takes.
    pub(crate) fn run_on_operation<A: Arch, M: Model, W: Workload>(
        &mut self,
        program: &mut Program<A, M, W>,
    ) {
        self.naively_allocate(program);
    }
}


#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Dd2;
    use crate::arch::Elements;
    use crate::formats::Bits;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::link::SendEnd;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::dataflow_ir::{GroupId, OpIndex, ProgramName, Units};
    use crate::islands::sentient::dialects::sentient::{ShuffleMode, Yielded};
    use crate::islands::sentient::{ProgramUnit, ProgramUnits};
    use crate::model::Model;
    use crate::units::DfirUnit;
    use crate::workload::Workload;

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

    /// `#sentient<reg_type L>` with no index yet — what every emitter before this pass writes.
    fn unassigned(locale: RegType) -> Reg {
        Reg {
            locale,
            index: None,
        }
    }

    /// `%r = sentient.scalar_add %a, %b {regLocale = locale}`.
    fn scalar_add(result: u32, reg: Option<Reg>) -> Op {
        Op::Sentient(sentient::Op::ScalarAdd {
            lhs: Val(1),
            rhs: Val(2),
            result: Val(result),
            reg,
            element_size: None,
            ty: ScalarTy::Index,
        })
    }

    /// `%r = sentient.load_and_send .. {regLocale = locale}` — an ADDRESS register, to show which
    /// counters a program unit does not restart.
    fn load_and_send(result: u32, locale: RegType) -> Op {
        Op::Sentient(sentient::Op::LoadAndSend {
            mutable_addr: Val(1),
            immutable_addr: Val(2),
            increment: Val(3),
            consumer: SendEnd::to_self(Val(98)),
            result: Val(result),
            extent: sentient::Extent {
                total_elements: Elements(64),
                element_size: Bits(16),
                chunk_size: Elements(1),
                chunk_stride: Elements(1),
                burst_size: Elements(1),
            },
            interleaved_group: Elements(0),
            rotate_val: None,
            dir: None,
            shuffle_mode: ShuffleMode::NoShuffle,
            reg: unassigned(locale),
            dbg_name: None,
        })
    }

    /// `sentient.for %iv = %bound {regLocales = [lccr], ..} { body }`, carrying nothing.
    fn counted_loop(iv: u32, body: Vec<Op>) -> Op {
        Op::Sentient(sentient::Op::For {
            iv: Val(iv),
            bound: Val(0),
            bound_reg: Some(unassigned(RegType::Lccr)),
            carried: Vec::new(),
            dbg_name: None,
            body,
        })
    }

    /// A module of `units`.
    fn program(units: Vec<Vec<Op>>) -> Program<Dd2, AnyModel, AnyRung> {
        let mut bodies = units.into_iter();
        let head = bodies.next().expect("one unit at least");
        let unit = |body: Vec<Op>| ProgramUnit {
            on: Units::one(DfirUnit::Lxlu, Val(0)),
            precision: None,
            body,
            arch: core::marker::PhantomData,
        };
        Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            preamble: Vec::new(),
            units: ProgramUnits::of(unit(head), bodies.map(unit).collect()),
            bound: core::marker::PhantomData,
        }
    }

    /// The register each op of `body` ended up with, in walk order.
    fn indices(body: &[Op]) -> Vec<Option<RegIndex>> {
        let mut out = Vec::new();
        for op in body {
            match op {
                Op::Sentient(sentient::Op::For { bound_reg, .. }) => {
                    out.push(bound_reg.and_then(|reg| reg.index));
                }
                Op::Sentient(sentient::Op::ScalarAdd { reg, .. }) => {
                    out.push(reg.and_then(|reg| reg.index));
                }
                Op::Sentient(sentient::Op::LoadAndSend { reg, .. }) => out.push(reg.index),
                Op::Sentient(sentient::Op::If { yielded, .. }) => {
                    out.extend(yielded.iter().map(|slot| slot.reg.index));
                }
                _ => {}
            }
            for region in dialects::regions_ref(op) {
                out.extend(indices(region));
            }
        }
        out
    }

    /// e344 — the vendor's own `regIndices = [0 : i32], regLocales = [lccr]` on a loop carrying
    /// nothing, LCCR as the nesting depth, and the per-unit restart that spares the address files.
    #[test]
    fn e344_naively_allocate() {
        let inner = counted_loop(11, vec![Op::Sentient(sentient::Op::Yield {
            results: Vec::new(),
        })]);
        let outer = counted_loop(
            10,
            vec![
                inner,
                Op::Sentient(sentient::Op::Yield {
                    results: Vec::new(),
                }),
            ],
        );
        let first = vec![
            outer,
            // The outer loop gave its LCCR back, so this loop is at depth 0 again.
            counted_loop(12, vec![Op::Sentient(sentient::Op::Yield {
                results: Vec::new(),
            })]),
            scalar_add(20, Some(unassigned(RegType::Lrf))),
            load_and_send(21, RegType::Lbr),
        ];
        let second = vec![
            scalar_add(30, Some(unassigned(RegType::Lrf))),
            load_and_send(31, RegType::Lbr),
        ];
        let mut module = program(vec![first, second]);

        let mut pass = RegisterAllocation::default();
        pass.naively_allocate(&mut module);

        let mut walked = module.units.iter();
        assert_eq!(
            indices(&walked.next().expect("the head unit").body),
            vec![
                // outer, inner, the second loop, the add, the load_and_send
                Some(RegIndex::at::<0>()),
                Some(RegIndex::at::<1>()),
                Some(RegIndex::at::<0>()),
                Some(RegIndex::at::<0>()),
                Some(RegIndex::at::<0>()),
            ]
        );
        assert_eq!(
            indices(&walked.next().expect("the second unit").body),
            vec![
                // ⭐ THE POINT: `lrf` restarted at the program unit and `lbr` did not.
                Some(RegIndex::at::<0>()),
                Some(RegIndex::at::<1>()),
            ]
        );
        assert_eq!(pass.failures(), &[]);
    }

    /// e344's negative — an `if` asking for an LRF is the `emitError` arm, and the WHOLE op keeps its
    /// unassigned registers even though the first entry was acceptable.
    #[test]
    fn e344_refuses_an_unknown_locale_and_leaves_the_whole_op_unassigned() {
        let branch = Op::Sentient(sentient::Op::If {
            predicate: sentient::CmpPredicate::Eq,
            lhs: Val(1),
            rhs: Val(2),
            yielded: vec![
                Yielded {
                    result: Val(3),
                    reg: unassigned(RegType::Jcr),
                    element_size: None,
                },
                Yielded {
                    result: Val(4),
                    reg: unassigned(RegType::Lrf),
                    element_size: None,
                },
            ],
            dbg_name: None,
            then_body: Vec::new(),
            else_body: Vec::new(),
        });
        let mut module = program(vec![vec![branch]]);

        let mut pass = RegisterAllocation::default();
        pass.naively_allocate(&mut module);

        let body = &module.units.iter().next().expect("the head unit").body;
        assert_eq!(indices(body), vec![None, None]);
        assert_eq!(
            pass.failures(),
            &[UnknownLocale {
                at: Val(4),
                locale: Some(RegType::Lrf),
                message: "Only JCR/XRFRDPTR/XRFWRPTR locales are accepted for if-ops",
            }]
        );
        // ⛔ AND THE JCR IT ALREADY TOOK STAYS TAKEN, which is what the reference's local vector does.
        assert_eq!(pass.counters.jcr, NextIndex(1));
    }

    /// e457 — the pass entry allocates, ungated: what `run_on_operation` leaves is what
    /// `naively_allocate` leaves.
    #[test]
    fn e457_run_on_operation() {
        let body = vec![
            scalar_add(20, Some(unassigned(RegType::Lrf))),
            scalar_add(21, Some(unassigned(RegType::Lrf))),
        ];
        let mut module = program(vec![body]);

        let mut pass = RegisterAllocation::default();
        pass.run_on_operation(&mut module);

        assert_eq!(
            indices(&module.units.iter().next().expect("the head unit").body),
            vec![Some(RegIndex::at::<0>()), Some(RegIndex::at::<1>())]
        );
        assert_eq!(pass.failures(), &[]);
    }
}
