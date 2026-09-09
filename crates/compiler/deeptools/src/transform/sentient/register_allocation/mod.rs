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

#![allow(dead_code)]
// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET — `e457_runOnOperation` (level 2) is the entry that
// reaches this file, and everything below is reachable only from the tests until it lands. CI runs
// clippy with `-D warnings`. ⭐ REMOVE THIS WITH e457.

use crate::arch::Arch;
use crate::islands::sentient::dialects::sentient as ops;
use crate::islands::sentient::dialects::{self, Op, UniformRegions, Val};
use crate::islands::sentient::{Program, ProgramUnit};
use crate::model::Model;
use crate::workload::Workload;

/// THE NINE `int` COUNTERS, ONE PER REGISTER FILE (`RegisterAllocation.cpp:40-48`).
///
/// ⛔ SIGNED, AND ONE OF THEM GOES DOWN: a `sentient.yield` closing a `sentient.for` decrements
/// `lccr_counter` (`:94-99`), so a loop whose slots took no LCCR leaves it at -1 — which is the
/// reference's own "no register" and [`minted`]'s `None`.
/// ⚠️ TRAP: `ebr_counter` IS DECLARED AND NEVER READ (`:48`). No arm places an EBR, so an EBR-localed
/// op keeps whatever index it arrived with; the field is here because the reference has it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Counters {
    lccr: i32,
    jcr: i32,
    lrf: i32,
    xrf_rd: i32,
    xrf_wr: i32,
    lar: i32,
    lbr: i32,
    ear: i32,
    ebr: i32,
}

/// `counter++` — the index this op takes, and the counter left pointing past it.
fn advance(counter: &mut i32) -> i32 {
    let at = *counter;
    *counter += 1;
    at
}

impl Counters {
    /// `isa<dataflow::ProgramUnitOp>(op)` (`:52-54`) — the five per-unit files restart.
    ///
    /// ⛔ THE FOUR ADDRESSING COUNTERS ARE **NOT** RESET, so `lar`/`lbr`/`ear`/`ebr` keep counting
    /// across program units, which is the reference's line and not an oversight to tidy.
    fn enter_program_unit(&mut self) {
        self.lccr = 0;
        self.jcr = 0;
        self.lrf = 0;
        self.xrf_rd = 0;
        self.xrf_wr = 0;
    }

    /// A `sentient.for` slot: `lrf`, `jcr`, `lccr`, and the two XRF pointers (`:71-90`).
    ///
    /// ⛔ THE XRF POINTERS DO NOT ADVANCE. `xrf_rd_counter += 1` is commented out at every one of its
    /// sites — *"Already handled in the above passes"* — so every XRF-localed slot of a unit is handed
    /// the SAME index.
    fn take_for_slot(&mut self, locale: ops::RegType) -> Option<i32> {
        match locale {
            ops::RegType::Lrf => Some(advance(&mut self.lrf)),
            ops::RegType::Jcr => Some(advance(&mut self.jcr)),
            ops::RegType::Lccr => Some(advance(&mut self.lccr)),
            ops::RegType::XrfRdPtr => Some(self.xrf_rd),
            ops::RegType::XrfWrPtr => Some(self.xrf_wr),
            _ => None,
        }
    }

    /// A `sentient.if` result or a `uniform.uniformize_regions` result: `jcr` and the two XRF
    /// pointers, and nothing else (`:104-116`, `:132-146`).
    fn take_jump(&mut self, locale: ops::RegType) -> Option<i32> {
        match locale {
            ops::RegType::Jcr => Some(advance(&mut self.jcr)),
            ops::RegType::XrfRdPtr => Some(self.xrf_rd),
            ops::RegType::XrfWrPtr => Some(self.xrf_wr),
            _ => None,
        }
    }

    /// A `sentient.scalar_add`, `scalar_sub` or `scalar_copy`: `lrf`, `jcr` and the two XRF pointers
    /// (`:162-176`, `:181-195`, `:200-214`).
    fn take_scalar(&mut self, locale: ops::RegType) -> Option<i32> {
        match locale {
            ops::RegType::Lrf => Some(advance(&mut self.lrf)),
            ops::RegType::Jcr => Some(advance(&mut self.jcr)),
            ops::RegType::XrfRdPtr => Some(self.xrf_rd),
            ops::RegType::XrfWrPtr => Some(self.xrf_wr),
            _ => None,
        }
    }

    /// A `sentient.load_and_send` or `receive_and_store`: `lrf`, `lar` or `lbr` (`:217-238`).
    fn take_addressing(&mut self, locale: ops::RegType) -> Option<i32> {
        match locale {
            ops::RegType::Lrf => Some(advance(&mut self.lrf)),
            ops::RegType::Lar => Some(advance(&mut self.lar)),
            ops::RegType::Lbr => Some(advance(&mut self.lbr)),
            _ => None,
        }
    }

    /// A `sentient.load_and_extract_scalar` result, or a `load_compute_and_send`: `lrf` alone
    /// (`:265-271`, `:307-313`).
    fn take_lrf(&mut self, locale: ops::RegType) -> Option<i32> {
        match locale {
            ops::RegType::Lrf => Some(advance(&mut self.lrf)),
            _ => None,
        }
    }

    /// A `sentient.load_and_store` result: `lar` or `ear` (`:288-296`).
    fn take_load_and_store(&mut self, locale: ops::RegType) -> Option<i32> {
        match locale {
            ops::RegType::Lar => Some(advance(&mut self.lar)),
            ops::RegType::Ear => Some(advance(&mut self.ear)),
            _ => None,
        }
    }
}

/// One index off a counter — and `None` for a negative one, which is the `-1` the reference pushes and
/// the `.td` prints for a register nothing assigned.
fn minted(counter: i32) -> Option<ops::RegIndex> {
    u32::try_from(counter).ok().map(ops::RegIndex::counted)
}

/// `op->emitError(<message>)` + `signalPassFailure()` AS DATA — the same shape
/// [`crate::transform::sentient::enhanced_dead_variable_elimination`] settled on, and for the same
/// reason: `signalPassFailure` is a flag on the pass object, so the walk keeps going and a caller can
/// still say WHICH op disagreed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownLocale {
    /// What the reference emitted.
    pub message: &'static str,
    /// The locale it would not place.
    pub locale: ops::RegType,
}

/// `"Unknown locale at the add operation"` — ⚠️ AND IT IS ALSO THE `sentient.for` MESSAGE (`:89`),
/// which names the add operation for a slot of a loop. The reference's copy-paste, kept.
pub const UNKNOWN_AT_ADD: &str = "Unknown locale at the add operation";
/// `"Only JCR/XRFRDPTR/XRFWRPTR locales are accepted for if-ops"` (`:114-116`).
pub const ONLY_JUMP_FOR_IF: &str = "Only JCR/XRFRDPTR/XRFWRPTR locales are accepted for if-ops";
/// `"Only JCR/XRFRDPTR/XRFWRPTR locales are accepted for uniformizeRegionOps"` (`:144-146`).
pub const ONLY_JUMP_FOR_UNIFORMIZE_REGIONS: &str =
    "Only JCR/XRFRDPTR/XRFWRPTR locales are accepted for uniformizeRegionOps";
/// `"Unknown locale at the sub operation"` (`:193`).
pub const UNKNOWN_AT_SUB: &str = "Unknown locale at the sub operation";
/// `"Unknown locale at the copy operation"` (`:212`).
pub const UNKNOWN_AT_COPY: &str = "Unknown locale at the copy operation";
/// `"Unknown locale at the load_and_send operation"` (`:230`).
pub const UNKNOWN_AT_LOAD_AND_SEND: &str = "Unknown locale at the load_and_send operation";
/// `"Unknown locale at the receive_and_store operation"` (`:249`).
pub const UNKNOWN_AT_RECEIVE_AND_STORE: &str = "Unknown locale at the receive_and_store operation";
/// `"Unknown locale at the load_and_extract_scalar operation"` (`:269`).
pub const UNKNOWN_AT_LOAD_AND_EXTRACT_SCALAR: &str =
    "Unknown locale at the load_and_extract_scalar operation";
/// `"Unknown locale at the load_and_store operation"` (`:293`).
pub const UNKNOWN_AT_LOAD_AND_STORE: &str = "Unknown locale at the load_and_store operation";
/// `"Unknown locale at the load_compute_and_send operation"` (`:310`).
pub const UNKNOWN_AT_LOAD_COMPUTE_AND_SEND: &str =
    "Unknown locale at the load_compute_and_send operation";

/// `RegisterAllocationPass`'s OWN STATE — the nine counters, which live across the whole module walk,
/// and every `signalPassFailure()` it made.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RegisterAllocation {
    counters: Counters,
    failures: Vec<UnknownLocale>,
}

impl RegisterAllocation {
    /// Replaces: e344_naivelyAllocate
    ///
    /// Hands every register-carrying sentient op the next index of its own file, counting from zero in
    /// each program unit, in pre-order over the module (`:39-319`).
    ///
    /// ⛔ IT IS THE ONLY THING THAT MINTS AN INDEX, so a `None` left here is the `regIndex = -1` the
    /// backend refuses with `Register initialization out of boundary`.
    /// ⛔ TRAP: A REFUSED LOCALE ABANDONS THE WHOLE ARRAY. `emitError` + `return` leaves the walk's
    /// callback before `setRegIndicesAttr`, so a `for`/`if`/`load_and_store` with one bad slot keeps
    /// EVERY old index — while the counters it already advanced stay advanced.
    pub fn naively_allocate<A: Arch, M: Model, W: Workload>(
        &mut self,
        program: &mut Program<A, M, W>,
    ) {
        // ⭐ `use_empty()` ASKS THE WHOLE MODULE, and this pass writes register indices and nothing
        // else, so the answer cannot change under the walk and is taken once, up front.
        let unused = unused_for_results(program);
        self.allocate_scope(&mut program.preamble, &unused, InFor::No);
        for unit in program.units.iter_mut() {
            self.counters.enter_program_unit();
            self.allocate_scope(&mut unit.body, &unused, InFor::No);
        }
    }

    /// Every `signalPassFailure()` this run made, in walk order.
    #[must_use]
    pub fn failures(&self) -> &[UnknownLocale] {
        &self.failures
    }

    /// `module_op.walk<WalkOrder::PreOrder>` over one region: the op, then what it holds.
    fn allocate_scope(&mut self, scope: &mut [Op], unused: &[Val], parent: InFor) {
        for op in scope.iter_mut() {
            self.allocate_op(op, unused, parent);
            let nested = match op {
                Op::Sentient(ops::Op::For { .. }) => InFor::Yes,
                _ => InFor::No,
            };
            for region in dialects::regions_mut(op) {
                self.allocate_scope(region, unused, nested);
            }
        }
    }

    /// The walk's one callback — the reference's `else if` chain, in its order.
    ///
    /// ⭐ `Op::Uniform(uniform::Op::UniformizeRegions)` — the spelling whose regions hold the rung
    /// BELOW's ops — carries no sentient register array in this island, which is exactly the
    /// `unif_regions->hasAttr(getRegLocalesAttrName())` gate (`:129-131`) closed, so it is a faithful
    /// no-op rather than a `todo!`.
    fn allocate_op(&mut self, op: &mut Op, unused: &[Val], parent: InFor) {
        match op {
            // `isa<sentient::ForOp>` (`:55-93`) — the `1 + n + n` slot array in one go.
            Op::Sentient(ops::Op::For {
                iv_reg, carried, ..
            }) => {
                let results: Vec<Val> = carried.iter().map(|value| value.result).collect();
                let mut indices: Vec<Option<ops::RegIndex>> = Vec::new();
                let mut refused = None;
                for (at, slot) in ops::for_reg_slots_mut(iv_reg, carried).iter().enumerate() {
                    // `int result_index = i - getNumRegionIterArgs() - 1` (`:63`): a slot past the
                    // induction variable and the arguments is a RESULT, and an unused result is left
                    // with no register at all (`:64-69`).
                    if let Some(result) = at
                        .checked_sub(1 + results.len())
                        .and_then(|i| results.get(i))
                        && unused.contains(result)
                    {
                        indices.push(None);
                        continue;
                    }
                    match self.counters.take_for_slot(slot.locale) {
                        Some(index) => indices.push(minted(index)),
                        None => {
                            refused = Some(slot.locale);
                            break;
                        }
                    }
                }
                match refused {
                    Some(locale) => self.refuse(UNKNOWN_AT_ADD, locale),
                    // `for_op.setRegIndicesAttr(...)` (`:93`).
                    None => {
                        for (slot, index) in ops::for_reg_slots_mut(iv_reg, carried)
                            .into_iter()
                            .zip(indices)
                        {
                            slot.index = index;
                        }
                    }
                }
            }
            // `isa<sentient::YieldOp>` (`:94-99`) — one LCCR handed back when a loop closes.
            Op::Sentient(ops::Op::Yield { .. }) => {
                if parent == InFor::Yes {
                    self.counters.lccr -= 1;
                }
            }
            // `isa<sentient::IfOp>` (`:100-126`) — one slot per yielded value.
            Op::Sentient(ops::Op::If { yielded, .. }) => {
                self.allocate_slots(
                    yielded.iter().map(|value| value.reg.locale).collect(),
                    ONLY_JUMP_FOR_IF,
                    Counters::take_jump,
                    |allocated| {
                        for (value, index) in yielded.iter_mut().zip(allocated) {
                            value.reg.index = index;
                        }
                    },
                );
            }
            // `dyn_cast<uniform::UniformizeRegionsOp>` (`:127-159`) — the op's own optional pair,
            // indexed by result number.
            Op::UniformRegions(UniformRegions::UniformizeRegions { results, .. }) => {
                // `unif_regions->hasAttr(locale_attr_str)` (`:131`): all-unallocated is the absent
                // attribute, and the whole arm is skipped.
                if results
                    .iter()
                    .all(|result| result.reg == ops::Reg::UNALLOCATED)
                {
                    return;
                }
                self.allocate_slots(
                    results.iter().map(|result| result.reg.locale).collect(),
                    ONLY_JUMP_FOR_UNIFORMIZE_REGIONS,
                    Counters::take_jump,
                    |allocated| {
                        for (result, index) in results.iter_mut().zip(allocated) {
                            result.reg.index = index;
                        }
                    },
                );
            }
            // `isa<sentient::AddOp>`, `SubOp`, `CopyOp` (`:160-215`) — one register each.
            Op::Sentient(ops::Op::ScalarAdd { reg, .. }) => {
                self.allocate_one(reg.as_mut(), UNKNOWN_AT_ADD, Counters::take_scalar);
            }
            Op::Sentient(ops::Op::ScalarSub { reg, .. }) => {
                self.allocate_one(reg.as_mut(), UNKNOWN_AT_SUB, Counters::take_scalar);
            }
            Op::Sentient(ops::Op::ScalarCopy { reg, .. }) => {
                self.allocate_one(Some(reg), UNKNOWN_AT_COPY, Counters::take_scalar);
            }
            // `dyn_cast<sentient::LoadAndSendOp>` / `ReceiveAndStoreOp` (`:216-252`).
            Op::Sentient(ops::Op::LoadAndSend { reg, .. }) => {
                self.allocate_one(
                    Some(reg),
                    UNKNOWN_AT_LOAD_AND_SEND,
                    Counters::take_addressing,
                );
            }
            Op::Sentient(ops::Op::ReceiveAndStore { reg, .. }) => {
                self.allocate_one(
                    Some(reg),
                    UNKNOWN_AT_RECEIVE_AND_STORE,
                    Counters::take_addressing,
                );
            }
            // `dyn_cast<sentient::LoadAndExtractScalarOp>` (`:253-277`) — its two results, both LRF.
            //
            // ⭐ THE `DT_CHECK_MSG(reg_locales.size() == kNumRegisters)` (`:259-261`) IS A FACT ABOUT
            // THE ISLAND: the two locales are two fields beside the two results, so they cannot
            // disagree in number.
            Op::Sentient(ops::Op::LoadAndExtractScalar {
                addr_reg, data_reg, ..
            }) => {
                self.allocate_slots(
                    vec![addr_reg.locale, data_reg.locale],
                    UNKNOWN_AT_LOAD_AND_EXTRACT_SCALAR,
                    Counters::take_lrf,
                    |allocated| {
                        for (reg, index) in [addr_reg, data_reg].into_iter().zip(allocated) {
                            reg.index = index;
                        }
                    },
                );
            }
            // `dyn_cast<sentient::LoadAndStoreOp>` (`:278-302`) — its two results, LAR or EAR.
            Op::Sentient(ops::Op::LoadAndStore {
                src_reg, dst_reg, ..
            }) => {
                self.allocate_slots(
                    vec![src_reg.locale, dst_reg.locale],
                    UNKNOWN_AT_LOAD_AND_STORE,
                    Counters::take_load_and_store,
                    |allocated| {
                        for (reg, index) in [src_reg, dst_reg].into_iter().zip(allocated) {
                            reg.index = index;
                        }
                    },
                );
            }
            // `dyn_cast<sentient::LoadComputeAndSendOp>` (`:303-317`).
            Op::Sentient(ops::Op::LoadComputeAndSend { reg, .. }) => {
                self.allocate_one(
                    Some(reg),
                    UNKNOWN_AT_LOAD_COMPUTE_AND_SEND,
                    Counters::take_lrf,
                );
            }
            // ⭐ EVERY OTHER OP FALLS OFF THE END OF THE `else if` CHAIN AND IS UNTOUCHED, including
            // `sentient.scalar_mul`, which has no `regIndex` at all (`SentientOps.td:816-829`).
            _ => {}
        }
    }

    /// One op, one register: `setRegIndexAttr(<counter>)` or the refusal.
    ///
    /// ⛔ `None` — AN OP WHOSE `regLocale` IS UNSAID — TAKES THE REFUSAL, because `getRegLocale()`
    /// answers `unknown` there and `unknown` is in no arm's accepted set.
    fn allocate_one(
        &mut self,
        reg: Option<&mut ops::Reg>,
        message: &'static str,
        take: fn(&mut Counters, ops::RegType) -> Option<i32>,
    ) {
        let locale = reg.as_ref().map_or(ops::RegType::Unknown, |reg| reg.locale);
        match take(&mut self.counters, locale) {
            Some(index) => {
                if let Some(reg) = reg {
                    reg.index = minted(index);
                }
            }
            None => self.refuse(message, locale),
        }
    }

    /// An op with a `regIndices` ARRAY: every locale placed, or — on the first refusal — the array
    /// left exactly as it was, which is the reference's `return` before its `setRegIndicesAttr`.
    fn allocate_slots(
        &mut self,
        locales: Vec<ops::RegType>,
        message: &'static str,
        take: fn(&mut Counters, ops::RegType) -> Option<i32>,
        write: impl FnOnce(Vec<Option<ops::RegIndex>>),
    ) {
        let mut allocated = Vec::with_capacity(locales.len());
        for locale in locales {
            match take(&mut self.counters, locale) {
                Some(index) => allocated.push(minted(index)),
                None => {
                    self.refuse(message, locale);
                    return;
                }
            }
        }
        write(allocated);
    }

    /// `op->emitError(message); signalPassFailure();`
    fn refuse(&mut self, message: &'static str, locale: ops::RegType) {
        self.failures.push(UnknownLocale { message, locale });
    }
}

/// WHETHER THE REGION BEING WALKED BELONGS TO A `sentient.for` — the one thing the `sentient.yield`
/// arm asks (`isa<sentient::ForOp>(yield_op->getParentRegion()->getParentOp())`, `:97`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InFor {
    Yes,
    No,
}

/// Every `sentient.for` RESULT no operand in the module reads — `getResult(i).use_empty()` (`:65`),
/// asked before the walk starts.
fn unused_for_results<A: Arch, M: Model, W: Workload>(program: &Program<A, M, W>) -> Vec<Val> {
    let mut results = Vec::new();
    collect_for_results(&program.preamble, &mut results);
    for unit in program.units.iter() {
        collect_for_results(&unit.body, &mut results);
    }
    results.retain(|val| use_empty(program, *val));
    results
}

/// The `sentient.for` results of one region and everything nested in it.
fn collect_for_results(scope: &[Op], results: &mut Vec<Val>) {
    for op in scope {
        if let Op::Sentient(ops::Op::For { carried, .. }) = op {
            results.extend(carried.iter().map(|value| value.result));
        }
        for region in dialects::regions_ref(op) {
            collect_for_results(region, results);
        }
    }
}

/// `Value::use_empty()` over the whole module.
fn use_empty<A: Arch, M: Model, W: Workload>(program: &Program<A, M, W>, val: Val) -> bool {
    dialects::use_count(val, &program.preamble) == 0
        && program
            .units
            .iter()
            .all(|unit: &ProgramUnit<A>| dialects::use_count(val, &unit.body) == 0)
}

// crustify:todo: e457_runOnOperation
//   authority : dcc/src/Transform/Sentient/RegisterAllocation.cpp:321  (1 body lines, level 2)
//   original  : void RegisterAllocationPass::runOnOperation()
//   calls     : e344_naivelyAllocate

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Dd2;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::dataflow_ir::{GroupId, OpIndex, ProgramName, Units};
    use crate::islands::sentient::ProgramUnits;
    use crate::islands::sentient::dialects::sentient::{Carried, Reg, RegType};
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

    /// A two-unit program, so the per-unit reset is observable.
    fn program(first: Vec<Op>, second: Vec<Op>) -> Program<Dd2, AnyModel, AnyRung> {
        Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            preamble: Vec::new(),
            units: ProgramUnits::of(
                ProgramUnit {
                    on: Units::one(DfirUnit::Lxlu, Val(0)),
                    iter_arg: None,
                    precision: None,
                    body: first,
                    arch: core::marker::PhantomData,
                },
                vec![ProgramUnit {
                    on: Units::one(DfirUnit::Lxsu, Val(1)),
                    iter_arg: None,
                    precision: None,
                    body: second,
                    arch: core::marker::PhantomData,
                }],
            ),
            bound: core::marker::PhantomData,
        }
    }

    /// `%out = sentient.scalar_add %a, %b {regLocale = locale}`.
    fn add(result: u32, locale: RegType) -> Op {
        Op::Sentient(ops::Op::ScalarAdd {
            lhs: Val(0),
            rhs: Val(0),
            result: Val(result),
            reg: Some(Reg {
                locale,
                index: None,
            }),
            element_size: None,
            ty: ScalarTy::Index,
        })
    }

    /// A `sentient.for` over `carried`, with an LCCR induction variable and `body`.
    fn for_op(carried: Vec<Carried>, body: Vec<Op>) -> Op {
        Op::Sentient(ops::Op::For {
            iv: Val(90),
            iv_reg: Reg {
                locale: RegType::Lccr,
                index: None,
            },
            bound: Val(91),
            carried,
            dbg_name: None,
            body,
        })
    }

    /// One carried value: an LRF argument slot and an LRF result slot.
    fn carried(arg: u32, result: u32) -> Carried {
        Carried {
            init: Val(0),
            arg: Val(arg),
            result: Val(result),
            reg: Reg {
                locale: RegType::Lrf,
                index: None,
            },
            result_reg: Reg {
                locale: RegType::Lrf,
                index: None,
            },
            program_header: false,
            element_size: None,
        }
    }

    /// e344 — the vendor's shape: each file counts from zero, the two XRF pointers hand out the SAME
    /// index twice, the loop's slots are `[iv, arg, result]` with an unread result left unassigned,
    /// its yield gives the LCCR back, and the second program unit starts over.
    #[test]
    fn e344_naively_allocate() {
        let mut program = program(
            vec![
                add(1, RegType::Lrf),
                add(2, RegType::Lrf),
                add(3, RegType::XrfRdPtr),
                add(4, RegType::XrfRdPtr),
                for_op(
                    vec![carried(10, 11)],
                    vec![Op::Sentient(ops::Op::Yield {
                        results: vec![Val(10)],
                    })],
                ),
                for_op(Vec::new(), Vec::new()),
            ],
            vec![add(6, RegType::Lrf)],
        );
        let mut pass = RegisterAllocation::default();
        pass.naively_allocate(&mut program);

        let mut units = program.units.iter();
        let first = &units.next().expect("the head unit").body;
        assert_eq!(index_of(&first[0]), Some(0));
        assert_eq!(index_of(&first[1]), Some(1));
        // ⭐ NO ADVANCE ON EITHER XRF POINTER — both take index 0.
        assert_eq!(index_of(&first[2]), Some(0));
        assert_eq!(index_of(&first[3]), Some(0));
        let Op::Sentient(ops::Op::For {
            iv_reg, carried, ..
        }) = &first[4]
        else {
            panic!("a sentient.for")
        };
        assert_eq!(iv_reg.index, Some(ops::RegIndex::at::<0>()));
        // The argument slot takes the third LRF; the result slot is read nowhere, so it takes none.
        assert_eq!(carried[0].reg.index, Some(ops::RegIndex::at::<2>()));
        assert_eq!(carried[0].result_reg.index, None);
        // ⭐ THE YIELD HANDED THE LCCR BACK, so the next loop's own counter is 0 again — and LCCR is
        // a loop's file alone: `take_scalar` refuses it, exactly as the reference's add arm does.
        let Op::Sentient(ops::Op::For { iv_reg, .. }) = &first[5] else {
            panic!("a sentient.for")
        };
        assert_eq!(iv_reg.index, Some(ops::RegIndex::at::<0>()));

        // The second unit's LRF counter restarted.
        let second = &units.next().expect("the tail unit").body;
        assert_eq!(index_of(&second[0]), Some(0));
        assert_eq!(pass.failures(), &[]);
    }

    /// e344 — a locale no arm accepts is one `signalPassFailure()`, and the op keeps no index.
    #[test]
    fn e344_refuses_a_locale_no_counter_covers() {
        let mut program = program(vec![add(1, RegType::Gtr)], Vec::new());
        let mut pass = RegisterAllocation::default();
        pass.naively_allocate(&mut program);
        assert_eq!(
            pass.failures(),
            &[UnknownLocale {
                message: UNKNOWN_AT_ADD,
                locale: RegType::Gtr,
            }]
        );
        let body = &program.units.iter().next().expect("the head unit").body;
        assert_eq!(index_of(&body[0]), None);
    }

    /// The `regIndex` a `sentient.scalar_add` ended up with, as a plain number.
    fn index_of(op: &Op) -> Option<u32> {
        let Op::Sentient(ops::Op::ScalarAdd { reg, .. }) = op else {
            panic!("a sentient.scalar_add")
        };
        reg.and_then(|reg| reg.index).map(|index| index.get())
    }
}
