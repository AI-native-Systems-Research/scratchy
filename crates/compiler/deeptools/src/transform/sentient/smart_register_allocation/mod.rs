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

//! `SmartRegisterAllocation.cpp` — 5 of the campaign's 656 units (dependency level(s) [0, 1, 2, 3]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e216_clean` | 216 | 0 | 4 | `dcc/src/Transform/Sentient/SmartRegisterAllocation.cpp:59` |
//! | `e217_isKnownToHaveSameValues` | 217 | 0 | 46 | `dcc/src/Transform/Sentient/SmartRegisterAllocation.cpp:87` |
//! | `e382_performGraphColoring` | 382 | 1 | 484 | `dcc/src/Transform/Sentient/SmartRegisterAllocation.cpp:136` |
//! | `e478_doRegisterAllocation` | 478 | 2 | 5 | `dcc/src/Transform/Sentient/SmartRegisterAllocation.cpp:81` |
//! | `e540_runOnOperation` | 540 | 3 | 8 | `dcc/src/Transform/Sentient/SmartRegisterAllocation.cpp:626` |

// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET — [`SmartRegisterAllocation::run_on_operation`]
// (e540) is the entry, and there is no ported D29-D75 pass driver to call it, so nothing but this
// file's own tests reaches anything here. CI runs clippy with `-D warnings`.
// ⭐ REMOVE THIS WITH THAT DRIVER, not with an anchor: e540 is filled and the pass is still unwired.
#![allow(dead_code)]

use std::collections::BTreeMap;

use super::analyses::{GreedyAllocator, Liveness, RegNode, RegisterGraphs};
use super::register_allocation::{NextIndex, UnknownLocale};
use crate::arch::Arch;
use crate::islands::sentient::dialects::sentient::{Reg, RegIndex, RegType};
use crate::islands::sentient::dialects::{
    self, Definitions, Op, UniformRegions, Val, sentient, uniform,
};
use crate::islands::sentient::{Program, ProgramUnit};
use crate::model::Model;
use crate::workload::Workload;

/// THE ALLOCATOR'S OWN STATE (`:54-55`) — the graphs it colours and the assignment it hands back.
///
/// ⛔ `register_assignment_` IS DEAD IN THE AUTHORITY TREE: nothing reads or writes it, and the
/// `RegAssignment()` (`:70`) that would have is declared and never defined. It is represented because
/// [`SmartRegisterAllocation::clean`] clears it, and clearing it is half of e216.
///
/// ⭐ GENERIC OVER THE OUT-OF-SCOPE GRAPHS, as [`super::analyses::RegisterGraphs`] requires — the
/// reference owns a concrete `RegisterGraphs`, and a parameter is what lets a test observe `clean()`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct SmartRegisterAllocation<G: RegisterGraphs> {
    /// `reg_graphs_`.
    pub(crate) reg_graphs: G,
    /// `register_assignment_` — `DenseMap<Value, int>`, the register each value was given.
    pub(crate) register_assignment: Vec<(Val, sentient::RegIndex)>,
    /// `signalPassFailure()` AS DATA — ⭐ NOT CLEARED BY [`Self::clean`], because MLIR's failure flag
    /// lives on the pass and e216 touches only the two members the reference names.
    pub(crate) failures: Vec<UnknownLocale>,
}

impl<G: RegisterGraphs> SmartRegisterAllocation<G> {
    /// Replaces: e216_clean
    ///
    /// Drops both halves of the allocator's state before the next program unit.
    ///
    /// ⛔ TRAP: NOT A RESET. `RegisterGraphs::clean` leaves `ec_map_` standing, so the same-colour
    /// equivalence classes cross the unit boundary — see [`RegisterGraphs::clean`].
    pub(crate) fn clean(&mut self) {
        self.reg_graphs.clean();
        self.register_assignment.clear();
    }
}

/// Replaces: e217_isKnownToHaveSameValues
///
/// Whether one of the two values is the result of a self-addressed transfer over the other: a
/// `load_and_send`, `receive_and_store` or `load_compute_and_send` whose increment is the constant 0
/// and whose `mutable_addr` IS the other value. Symmetric, `op_1` tested first.
///
/// ⛔ TRAP: THE REFERENCE DEREFERENCES A NULL HERE. `getIncrement().getDefiningOp<ConstantOp>()
/// .getValue()` is unguarded (`:90-92`), so an increment that is not a `sentient.scalar_constant`
/// crashes it; this answers `false`, which is the arm the reference would have taken had it checked.
/// ⛔ AND IT HAS NO CALLER in the authority tree — declared at `:63`, defined at `:87`, called nowhere.
#[must_use]
pub(crate) fn is_known_to_have_same_values(op_1: Val, op_2: Val, defs: Definitions<'_>) -> bool {
    addresses(op_1, op_2, defs) || addresses(op_2, op_1, defs)
}

/// One direction of [`is_known_to_have_same_values`] — `addr` is `value`'s own `mutable_addr` and the
/// transfer never advances it.
fn addresses(value: Val, addr: Val, defs: Definitions<'_>) -> bool {
    let Some(Op::Sentient(op)) = defs.of(value) else {
        return false;
    };
    let (mutable_addr, increment) = match op {
        sentient::Op::LoadAndSend {
            mutable_addr,
            increment,
            ..
        }
        | sentient::Op::ReceiveAndStore {
            mutable_addr,
            increment,
            ..
        }
        | sentient::Op::LoadComputeAndSend {
            mutable_addr,
            increment,
            ..
        } => (*mutable_addr, *increment),
        _ => return false,
    };
    matches!(
        defs.of(increment),
        Some(Op::Sentient(sentient::Op::ScalarConstant { value: 0, .. }))
    ) && mutable_addr == addr
}

/// `useGreedyAllocator = (opts_.OptLevel == 0)`, set once in the pass constructor (`:48`).
///
/// ⛔ NOT A `dcc-opt` FLAG BUT A BUILD OPTION: `CommonPassOptions::OptLevel` defaults to `-1`
/// (`dcc/tools/Options/dcc-pass-option.h:117`), so an ordinary build colours normally and only `-O0`
/// is greedy. The `Passes.td` option of the same name (`Passes.td:365`) is overwritten by that
/// constructor line and never read.
const USE_GREEDY_ALLOCATOR: GreedyAllocator = GreedyAllocator::No;

/// `<locale>0` — what a `std::map` subscript reads for a node no colouring placed, and the index the
/// two xrf pointers always get.
const ZERO: RegIndex = RegIndex::at::<0>();

/// THE EIGHT LOCALES e382 COLOURS, IN THE ORDER IT COLOURS THEM (`:139-179`).
///
/// ⛔ THE ORDER IS OBSERVABLE: each triple folds `liveness`'s virtual assignments into the SHARED
/// `ec_map_` that [`RegisterGraphs::clean`] does not clear, so a later locale sees the earlier ones'
/// equivalence classes.
const COLOURED_LOCALES: [RegType; 8] = [
    RegType::Lrf,
    RegType::Jcr,
    RegType::Lar,
    RegType::Lbr,
    RegType::Ear,
    RegType::Ebr,
    RegType::Gtr,
    RegType::Mvr,
];

/// THE EIGHT COLOURINGS, keyed by locale rather than named one `std::map<int, int>` local each
/// (`:142-179`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Colorings(BTreeMap<RegType, BTreeMap<RegNode, RegIndex>>);

impl Colorings {
    /// `<locale>_coloring[graph_index]`, and `None` where no arm of the colouring loop covers
    /// `locale`.
    ///
    /// ⛔ TRAP: `std::map::operator[]` INSERTS A ZERO. A node the colouring never placed reads back as
    /// register 0 instead of being refused, so a value liveness never indexed silently lands in
    /// `<locale>0` — and [`Liveness::operand_to_index`] carries the same trap one level down.
    fn index(&self, locale: RegType, node: RegNode) -> Option<RegIndex> {
        Some(self.0.get(&locale)?.get(&node).copied().unwrap_or(ZERO))
    }
}

/// WHICH REGISTER FILES ONE OP FORM ACCEPTS — the `if / else if` chain each arm of the walk spells out.
///
/// ⛔ NOT [`super::register_allocation`]'s SETS OF THE SAME NAME: the naive allocator's chains differ
/// op for op (no GTR loop, no LRF branch, no MVR copy, LBR instead of EAR on a transfer), so the two
/// enums are two different facts about two different passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Accepts {
    /// `sentient.for` (`:191-256`) — the only arm that takes LCCR.
    Loop,
    /// `sentient.if` (`:268-307`) and `uniform.uniformize_regions` (`:319-356`).
    Branch,
    /// `sentient.scalar_add` (`:368-397`) and `scalar_sub` (`:401-430`).
    Scalar,
    /// `sentient.scalar_copy` (`:434-483`) — the widest chain, and the only one taking EBR or MVR.
    Copy,
    /// `sentient.load_and_send` (`:487-508`) and `receive_and_store` (`:552-573`).
    Transfer,
    /// `sentient.load_and_extract_scalar` (`:513-533`), and the required locale of the two
    /// `DT_CHECK`ed ops (`:580`, `:613`).
    LrfOnly,
    /// `sentient.load_and_store` (`:588-599`).
    Address,
}

impl Accepts {
    /// Whether this op form's chain has an arm for `locale`.
    fn takes(self, locale: RegType) -> bool {
        match self {
            Accepts::Loop => matches!(
                locale,
                RegType::Lrf
                    | RegType::Jcr
                    | RegType::Lar
                    | RegType::Ear
                    | RegType::Gtr
                    | RegType::Lccr
                    | RegType::XrfRdPtr
                    | RegType::XrfWrPtr
            ),
            Accepts::Branch => matches!(
                locale,
                RegType::Jcr
                    | RegType::Lar
                    | RegType::Ear
                    | RegType::Gtr
                    | RegType::XrfRdPtr
                    | RegType::XrfWrPtr
                    | RegType::Lrf
            ),
            Accepts::Scalar => matches!(
                locale,
                RegType::Lrf
                    | RegType::Jcr
                    | RegType::Lar
                    | RegType::Ear
                    | RegType::XrfRdPtr
                    | RegType::XrfWrPtr
            ),
            Accepts::Copy => matches!(
                locale,
                RegType::Lrf
                    | RegType::Jcr
                    | RegType::Lar
                    | RegType::Ear
                    | RegType::Lbr
                    | RegType::Ebr
                    | RegType::Gtr
                    | RegType::XrfRdPtr
                    | RegType::XrfWrPtr
                    | RegType::Mvr
            ),
            Accepts::Transfer => matches!(locale, RegType::Lrf | RegType::Lar | RegType::Ear),
            Accepts::LrfOnly => matches!(locale, RegType::Lrf),
            Accepts::Address => matches!(locale, RegType::Ear | RegType::Lar),
        }
    }
}

/// `"Unknown locale at the for operation"` (`:256`).
const UNKNOWN_AT_FOR: &str = "Unknown locale at the for operation";
/// `:307-309`, the two string literals concatenated as the compiler joins them.
const ONLY_FOR_IF: &str =
    "Only JCR/LRF/LAR/EAR/GTR/XRFRDPTR/XRFWRPTR locales are accepted for if-ops";

/// WHOSE REGION THE WALK IS IN — the one question this pass asks about a parent (`:265`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Parent {
    /// A `sentient.for` body, where a `sentient.yield` gives the LCCR index back.
    For,
    /// Anything else, including the program unit body itself.
    Other,
}

/// THE LOCALS ONE `performGraphColoring` CARRIES THROUGH ITS WALK (`:182-188`).
struct Assign<'a, L: Liveness> {
    /// The eight colourings, already computed.
    colorings: Colorings,
    /// `liveness`, asked only for `getOperandToIndex()`.
    liveness: &'a mut L,
    /// `lccr_counter` — ⭐ THE LOOP NESTING DEPTH, by the `for`/`yield` pair, and the ONLY counter here
    /// that is read before it is written.
    lccr: NextIndex,
    /// Every `emitError` + `signalPassFailure` the walk made.
    failures: Vec<UnknownLocale>,
}

impl<L: Liveness> Assign<'_, L> {
    /// `unit.walk<WalkOrder::PreOrder>` (`:190`) — the op, then its regions.
    fn walk(&mut self, scope: &mut [Op], parent: Parent) {
        for op in scope.iter_mut() {
            self.assign_to(op, parent);
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

    /// One op of the walk — the `if / else if` chain over op kinds (`:191-620`).
    fn assign_to(&mut self, op: &mut Op, parent: Parent) {
        match op {
            Op::Sentient(sentient::Op::For {
                iv,
                bound_reg,
                carried,
                ..
            }) => {
                // ⭐ ENTRY 0 IS THE INDUCTION VARIABLE: `curr_iter_args[i - 1]` at `i == 0` reads
                // `getRegionIterArgs()`, which is `arguments().drop_front(1)`, one before its start —
                // body argument 0, the iv.
                let asked: Vec<(Val, RegType)> = bound_reg
                    .as_ref()
                    .map(|reg| (*iv, reg.locale))
                    .into_iter()
                    .chain(carried.iter().map(|slot| (slot.arg, slot.reg.locale)))
                    .collect();
                let Some(indices) = self.indices_for(&asked, Accepts::Loop, UNKNOWN_AT_FOR) else {
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
                // `isa<sentient::ForOp>(yield_op->getParentRegion()->getParentOp())` (`:265`).
                if matches!(parent, Parent::For) {
                    self.lccr.release();
                }
            }
            Op::Sentient(sentient::Op::If { yielded, .. }) => {
                let asked: Vec<(Val, RegType)> = yielded
                    .iter()
                    .map(|slot| (slot.result, slot.reg.locale))
                    .collect();
                let Some(indices) = self.indices_for(&asked, Accepts::Branch, ONLY_FOR_IF) else {
                    return;
                };
                for (slot, index) in yielded.iter_mut().zip(indices) {
                    slot.reg.index = Some(index);
                }
            }
            // `dyn_cast<uniform::UniformizeRegionsOp>(op)` (`:318-367`), whose whole body is inside
            // `if (const auto locales = unif_regions.getRegLocalesAttr(); locales)`.
            //
            // ⛔ NEITHER ISLAND SPELLING CARRIES THAT ATTRIBUTE, for the reason
            // [`super::register_allocation`]'s own uniformize arm records: `uniform` at this rung IS the
            // DataflowIR dialect, and the sentient-rung [`UniformRegions`] spelling has no printer for
            // the register arrays either. Every `uniform.uniformize_regions` here is therefore the
            // reference's own no-attribute case, which assigns nothing and refuses nothing.
            Op::Uniform(uniform::Op::UniformizeRegions { .. })
            | Op::UniformRegions(UniformRegions::UniformizeRegions { .. }) => {}
            Op::Sentient(sentient::Op::ScalarAdd { result, reg, .. }) => {
                let message = "Unknown locale at the add operation";
                self.assign_singular(*result, reg.as_mut(), Accepts::Scalar, message);
            }
            Op::Sentient(sentient::Op::ScalarSub { result, reg, .. }) => {
                let message = "Unknown locale at the sub operation";
                self.assign_singular(*result, reg.as_mut(), Accepts::Scalar, message);
            }
            Op::Sentient(sentient::Op::ScalarCopy { result, reg, .. }) => {
                let message = "Unknown locale at the copy operation";
                self.assign_singular(*result, Some(reg), Accepts::Copy, message);
            }
            Op::Sentient(sentient::Op::LoadAndSend { result, reg, .. }) => {
                let message = "Unknown locale at the load_and_send operation";
                self.assign_singular(*result, Some(reg), Accepts::Transfer, message);
            }
            Op::Sentient(sentient::Op::ReceiveAndStore { result, reg, .. }) => {
                let message = "Unknown locale at the receive_and_store operation";
                self.assign_singular(*result, Some(reg), Accepts::Transfer, message);
            }
            Op::Sentient(sentient::Op::LoadAndExtractScalar {
                addr_result,
                data_result,
                addr_reg,
                data_reg,
                ..
            }) => {
                // ⭐ BOTH LOCALES ARE CHECKED BEFORE EITHER INDEX IS READ (`:517-537`), and the DATA
                // half is checked first even though it is written to entry 1 of the array.
                if !Accepts::LrfOnly.takes(data_reg.locale) {
                    self.failures.push(UnknownLocale {
                        at: *data_result,
                        locale: Some(data_reg.locale),
                        message: "Unknown locale at the load_and_extract operation for data result",
                    });
                    return;
                }
                if !Accepts::LrfOnly.takes(addr_reg.locale) {
                    self.failures.push(UnknownLocale {
                        at: *addr_result,
                        locale: Some(addr_reg.locale),
                        message: "Unknown locale at the load_and_extract operation for addr result",
                    });
                    return;
                }
                let data_index = self.lrf_index(*data_result);
                if data_index != ZERO {
                    panic!(
                        "DT_CHECK_MSG(lrf_coloring[graph_index] == 0, \"only LRF0 supported for \
                         load_and_extract_scalar op data results\") \
                         (`SmartRegisterAllocation.cpp:541`)"
                    );
                }
                data_reg.index = Some(data_index);
                addr_reg.index = Some(self.lrf_index(*addr_result));
            }
            Op::Sentient(sentient::Op::ReceiveAndExtractScalar { result, reg, .. }) => {
                if !Accepts::LrfOnly.takes(reg.locale) {
                    panic!(
                        "DT_CHECK_MSG(receive_and_extract_op.getRegLocale() == \
                         SentientRegType::lrf, \"ReceiveAndExtractScalarOp must be LRF locale\") \
                         (`SmartRegisterAllocation.cpp:580`)"
                    );
                }
                reg.index = Some(self.lrf_index(*result));
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
                if !Accepts::LrfOnly.takes(reg.locale) {
                    panic!(
                        "DT_CHECK(load_compute_op.getRegLocale() == SentientRegType::lrf) \
                         (`SmartRegisterAllocation.cpp:613`)"
                    );
                }
                reg.index = Some(self.lrf_index(*result));
            }
            // ⭐ THE REFERENCE'S CHAIN HAS NO `else`: an op it names no arm for keeps whatever register
            // an earlier pass gave it.
            _ => {}
        }
    }

    /// `lrf_coloring[liveness.getOperandToIndex()[value]]` — the three `DT_CHECK`ed reads, which
    /// consult no chain because their locale is already known.
    fn lrf_index(&mut self, value: Val) -> RegIndex {
        let node = self.liveness.operand_to_index(value);
        self.colorings.index(RegType::Lrf, node).unwrap_or(ZERO)
    }

    /// The whole array, or nothing — see [`SmartRegisterAllocation::perform_graph_coloring`]'s TRAP.
    fn indices_for(
        &mut self,
        asked: &[(Val, RegType)],
        accepts: Accepts,
        message: &'static str,
    ) -> Option<Vec<RegIndex>> {
        let mut indices = Vec::with_capacity(asked.len());
        for (at, locale) in asked {
            indices.push(self.index_for(*at, *locale, accepts, message)?);
        }
        Some(indices)
    }

    /// `op.setRegIndexAttr(builder.getI32IntegerAttr(counter))` — the ops carrying ONE `regLocale`.
    ///
    /// ⛔ `None` IS THE ABSENT ATTRIBUTE, which `getRegLocale()` cannot answer any arm of the chain
    /// with, so it lands on the same `emitError`.
    fn assign_singular(
        &mut self,
        at: Val,
        reg: Option<&mut Reg>,
        accepts: Accepts,
        message: &'static str,
    ) {
        match reg {
            Some(reg) => {
                if let Some(index) = self.index_for(at, reg.locale, accepts, message) {
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

    /// The index `locale` asks for at `at`, and the recorded `emitError` + `signalPassFailure` where
    /// this op form's chain has no arm for it.
    fn index_for(
        &mut self,
        at: Val,
        locale: RegType,
        accepts: Accepts,
        message: &'static str,
    ) -> Option<RegIndex> {
        if accepts.takes(locale) {
            // ⛔ THREE LOCALES CONSULT NO COLOURING: LCCR counts loop depth, and the two xrf pointers
            // push a counter whose `+= 1` is commented out ("Already handled in the above passes"), so
            // every xrf pointer in the unit is index 0.
            let index = match locale {
                RegType::Lccr => Some(self.lccr.take()),
                RegType::XrfRdPtr | RegType::XrfWrPtr => Some(ZERO),
                _ => {
                    let node = self.liveness.operand_to_index(at);
                    self.colorings.index(locale, node)
                }
            };
            if let Some(index) = index {
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
}

impl<G: RegisterGraphs> SmartRegisterAllocation<G> {
    /// Replaces: e478_doRegisterAllocation
    ///
    /// One program unit's allocation: drop the previous unit's state, build this unit's interference
    /// graphs from its live ranges, then colour them.
    ///
    /// ⛔ ORDER IS THE PORT: `clean()` FIRST, so the graphs are built into an emptied allocator — and
    /// `RegisterGraphs::clean` leaving `ec_map_` standing is what makes that order observable.
    pub(crate) fn do_register_allocation<A: Arch, L: Liveness>(
        &mut self,
        unit: &mut ProgramUnit<A>,
        liveness: &mut L,
    ) {
        self.clean();
        // `buildGraphs(dccExtContext(), liveness, unit)` — all three trailing defaults left alone.
        self.reg_graphs
            .build_graphs(liveness, &unit.body, RegType::Unknown, None);
        self.perform_graph_coloring(unit, liveness, USE_GREEDY_ALLOCATOR);
    }

    /// Replaces: e382_performGraphColoring
    ///
    /// Colours all eight register files over this unit's hyper-graphs, then walks it in pre-order
    /// giving every register-carrying op the colour its own result got.
    ///
    /// ⛔ TRAP: A REFUSED ENTRY ABANDONS THE WHOLE OP, and an LCCR index already taken stays taken.
    /// ⭐ ONLY `lccr_counter` IS STATE (`:182-188`): the other nine are write-before-read at every
    /// use, and the two xrf pointers never advance, so every xrf pointer here is index 0. The
    /// colouring's own subscript trap is at [`Colorings::index`].
    pub(crate) fn perform_graph_coloring<A: Arch, L: Liveness>(
        &mut self,
        unit: &mut ProgramUnit<A>,
        liveness: &mut L,
        greedy: GreedyAllocator,
    ) {
        let mut colorings = Colorings::default();
        for locale in COLOURED_LOCALES {
            self.reg_graphs
                .create_same_color_edge_eq_classes(liveness, locale);
            self.reg_graphs.build_hyper_graph(locale);
            let coloring = self
                .reg_graphs
                .do_graph_color_on_locale(locale, &unit.body, greedy);
            colorings.0.insert(locale, coloring);
        }

        let mut assign = Assign {
            colorings,
            liveness,
            lccr: NextIndex::default(),
            failures: Vec::new(),
        };
        assign.walk(&mut unit.body, Parent::Other);
        self.failures.append(&mut assign.failures);
    }

    /// Every `emitError` + `signalPassFailure` this pass made.
    #[must_use]
    pub(crate) fn failures(&self) -> &[UnknownLocale] {
        &self.failures
    }
}

impl<G: RegisterGraphs> SmartRegisterAllocation<G> {
    /// Replaces: e540_runOnOperation
    ///
    /// The pass entry: allocate registers in every program unit of the module, each against its own
    /// live ranges.
    ///
    /// ⛔ NO DISABLE FLAG AND NO UNIT FILTER — unlike most passes in this campaign this one runs
    /// unconditionally over every unit, whatever it is on (`:626-634`).
    /// ⭐ `getChildAnalysis<Liveness>(unit)` IS A CONSTRUCTION: `Liveness(Operation *p) {
    /// computeRegisterLiveRange(p); }` (`Analyses/Liveness.h:85`), so a fresh analysis per unit,
    /// computed over that unit, is the port — the cache is MLIR's and holds nothing across units.
    /// ⚠️ `DEBUG_WITH_TYPE(VerboseDebug, liveness.dump())` IS DROPPED (`:630`): it changes no IR.
    pub(crate) fn run_on_operation<A: Arch, M: Model, W: Workload, L: Liveness + Default>(
        &mut self,
        program: &mut Program<A, M, W>,
    ) {
        for unit in program.units.iter_mut() {
            let mut liveness = L::default();
            liveness.compute_register_live_range(&unit.body);
            self.do_register_allocation(unit, &mut liveness);
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::{
        COLOURED_LOCALES, GreedyAllocator, Liveness, RegNode, RegisterGraphs,
        SmartRegisterAllocation, is_known_to_have_same_values,
    };
    use std::collections::BTreeMap;
    use crate::arch::{Dd2, Elements};
    use crate::formats::Bits;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::link::SendEnd;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::dataflow_ir::{GroupId, OpIndex, ProgramName, Units};
    use crate::islands::sentient::dialects::sentient::{Carried, Reg, RegIndex, RegType, ShuffleMode};
    use crate::islands::sentient::dialects::{Definitions, Op, Val, sentient};
    use crate::islands::sentient::{Program, ProgramUnit, ProgramUnits};
    use crate::model::Model;
    use crate::transform::sentient::analyses::VirtualAssigns;
    use crate::transform::sentient::local_region_splitting_for_value_commoning::MaxRegNum;
    use crate::units::DfirUnit;
    use crate::workload::Workload;
    use core::sync::atomic::{AtomicU32, Ordering};

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

    /// A one-op-list PE program unit — e478 reads nothing of it but its body.
    fn unit_of(body: Vec<Op>) -> ProgramUnit<Dd2> {
        ProgramUnit {
            on: Units::one(DfirUnit::Pe, Val(0)),
            precision: None,
            body,
            arch: core::marker::PhantomData,
        }
    }

    /// A `RegisterGraphs` THAT ONLY RECORDS BEING CLEANED — the analysis is out of campaign scope, so
    /// `clean()` is the whole of its observable surface.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    struct CountingGraphs {
        cleaned: u32,
        /// How many ops the graphs were last built over, and whether they had been cleaned first.
        built_over: Option<(usize, u32)>,
    }

    impl RegisterGraphs for CountingGraphs {
        fn clean(&mut self) {
            self.cleaned += 1;
        }

        fn build_graphs<L: Liveness>(
            &mut self,
            _liveness: &mut L,
            unit: &[Op],
            _locale: RegType,
            _coreunit: Option<Val>,
        ) {
            self.built_over = Some((unit.len(), self.cleaned));
        }

        fn create_same_color_edge_eq_classes<L: Liveness>(
            &mut self,
            _liveness: &mut L,
            _locale: RegType,
        ) {
            todo!("e478 stops at the first of e382's eight colouring triples")
        }

        fn build_hyper_graph(&mut self, _locale: RegType) {
            todo!("no unit here builds a hyper-graph through this fake")
        }

        fn fast_check_colorability(&mut self, _num_colors: MaxRegNum, _locale: RegType) -> bool {
            todo!("no unit here checks colorability through this fake")
        }

        fn do_graph_color_on_locale(
            &mut self,
            _locale: RegType,
            _unit: &[Op],
            _greedy: GreedyAllocator,
        ) -> BTreeMap<RegNode, RegIndex> {
            todo!("no unit here colours through this fake")
        }
    }

    /// A `Liveness` e478 ONLY PASSES ALONG — it asks it nothing itself, and the two graph calls that
    /// would are the out-of-scope analysis's.
    #[derive(Debug, Clone, Copy, Default)]
    struct SilentLiveness;

    impl Liveness for SilentLiveness {
        fn update_live_ranges_for_program_header_promotion(&mut self, _candidate: Val) {}

        fn is_live_range_overlaps(&self, _val1: Val, _val2: Val) -> bool {
            false
        }

        fn clear(&mut self, _virtual_assigns: VirtualAssigns) {
            todo!("no unit here clears this fake")
        }

        fn compute_register_live_range(&mut self, _unit: &[Op]) {
            todo!("no unit here recomputes this fake")
        }

        fn add_virtual_assign_optional(&mut self, _set_of_subsets: &[Vec<Val>]) {
            todo!("no unit here offers a subset to this fake")
        }

        fn add_virtual_assign_enforced(&mut self, _set_of_pairs: &[(Val, Val)]) {
            todo!("no unit here enforces a pair on this fake")
        }

        fn operand_to_index(&mut self, _value: Val) -> RegNode {
            todo!("e478 never reaches a colouring lookup through this fake")
        }
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

    /// `sentient.load_and_send` addressed by `mutable_addr` and advanced by `increment`.
    fn load_and_send(mutable_addr: u32, increment: u32, result: u32) -> Op {
        Op::Sentient(sentient::Op::LoadAndSend {
            mutable_addr: Val(mutable_addr),
            immutable_addr: Val(99),
            increment: Val(increment),
            consumer: SendEnd::to_self(Val(98)),
            result: Val(result),
            extent: sentient::Extent {
                total_elements: Elements(64),
                element_size: Bits(32),
                chunk_size: Elements(1),
                chunk_stride: Elements(1),
                burst_size: Elements(1),
            },
            interleaved_group: Elements(0),
            rotate_val: None,
            dir: None,
            shuffle_mode: ShuffleMode::NoShuffle,
            reg: Reg {
                locale: RegType::Lar,
                index: None,
            },
            dbg_name: None,
        })
    }

    /// e216 — both halves go: the graphs are cleaned and the assignment is emptied.
    #[test]
    fn e216_cleans_the_graphs_and_empties_the_assignment() {
        let mut pass = SmartRegisterAllocation {
            reg_graphs: CountingGraphs::default(),
            register_assignment: vec![(Val(7), sentient::RegIndex::at::<3>())],
            failures: Vec::new(),
        };

        pass.clean();

        assert_eq!(
            pass.reg_graphs,
            CountingGraphs {
                cleaned: 1,
                // e216 never builds, which is what makes e478's ordering assertion legible.
                built_over: None,
            }
        );
        assert_eq!(pass.register_assignment, Vec::new());
    }

    /// e217 — the vendor's own shape, symmetric in both argument orders, and the two ways it says no.
    #[test]
    fn e217_a_never_advancing_transfer_over_the_other_value_is_the_same_value() {
        let ops = vec![
            scalar_const(1, 0),
            scalar_const(2, 1),
            // `%20`'s transfer reads `%10` and never advances it.
            load_and_send(10, 1, 20),
            // `%21`'s advances by one.
            load_and_send(11, 2, 21),
        ];
        let regions: [&[Op]; 1] = [&ops];
        let defs = Definitions::from_innermost(&regions);

        assert!(is_known_to_have_same_values(Val(20), Val(10), defs));
        assert!(is_known_to_have_same_values(Val(10), Val(20), defs));
        assert!(!is_known_to_have_same_values(Val(21), Val(11), defs));
        assert!(!is_known_to_have_same_values(Val(20), Val(11), defs));
    }

    /// e478 — the allocator is cleaned BEFORE its graphs are built over the unit, and the colouring
    /// seam is then reached. ⭐ THE ORDER IS WHAT IS OBSERVABLE, and it is the whole of the port.
    #[test]
    fn e478_cleans_then_builds_the_graphs_then_colours() {
        let mut pass = SmartRegisterAllocation {
            reg_graphs: CountingGraphs::default(),
            register_assignment: vec![(Val(7), sentient::RegIndex::at::<3>())],
            failures: Vec::new(),
        };
        let mut unit = unit_of(vec![scalar_const(1, 0), scalar_const(2, 1)]);
        let mut liveness = SilentLiveness;

        let reached = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pass.do_register_allocation(&mut unit, &mut liveness);
        }));

        assert!(
            reached.is_err(),
            "the fake has no equivalence classes, so e382's first colouring triple panics"
        );
        assert_eq!(
            pass.reg_graphs,
            CountingGraphs {
                cleaned: 1,
                built_over: Some((2, 1)),
            }
        );
        assert_eq!(pass.register_assignment, Vec::new());
    }

    /// A `RegisterGraphs` THAT ANSWERS WITH A COLOURING — e382 is the one unit that reads one back, so
    /// this fake also records WHICH locales it was asked, in order.
    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    struct ColouringGraphs {
        asked: Vec<RegType>,
        greedy: Vec<GreedyAllocator>,
    }

    impl RegisterGraphs for ColouringGraphs {
        fn clean(&mut self) {}

        fn build_graphs<L: Liveness>(
            &mut self,
            _liveness: &mut L,
            _unit: &[Op],
            _locale: RegType,
            _coreunit: Option<Val>,
        ) {
        }

        fn create_same_color_edge_eq_classes<L: Liveness>(
            &mut self,
            _liveness: &mut L,
            _locale: RegType,
        ) {
        }

        fn build_hyper_graph(&mut self, _locale: RegType) {}

        fn fast_check_colorability(&mut self, _num_colors: MaxRegNum, _locale: RegType) -> bool {
            todo!("e382 never checks colorability")
        }

        fn do_graph_color_on_locale(
            &mut self,
            locale: RegType,
            _unit: &[Op],
            greedy: GreedyAllocator,
        ) -> BTreeMap<RegNode, RegIndex> {
            self.asked.push(locale);
            self.greedy.push(greedy);
            match locale {
                // The carried value `%11` got LRF2.
                RegType::Lrf => BTreeMap::from([(RegNode(11), RegIndex::at::<2>())]),
                // The copy's result `%30` got MVR5.
                RegType::Mvr => BTreeMap::from([(RegNode(30), RegIndex::at::<5>())]),
                _ => BTreeMap::new(),
            }
        }
    }

    /// A `Liveness` WHOSE GRAPH NODE IS THE VALUE'S OWN NUMBER — e382 asks it nothing else.
    #[derive(Debug, Clone, Copy, Default)]
    struct NumberedLiveness;

    impl Liveness for NumberedLiveness {
        fn update_live_ranges_for_program_header_promotion(&mut self, _candidate: Val) {}

        fn is_live_range_overlaps(&self, _val1: Val, _val2: Val) -> bool {
            false
        }

        fn clear(&mut self, _virtual_assigns: VirtualAssigns) {
            todo!("e382 never clears liveness")
        }

        fn compute_register_live_range(&mut self, _unit: &[Op]) {
            todo!("e382 is handed a computed liveness")
        }

        fn add_virtual_assign_optional(&mut self, _set_of_subsets: &[Vec<Val>]) {
            todo!("e382 adds no virtual assignment")
        }

        fn add_virtual_assign_enforced(&mut self, _set_of_pairs: &[(Val, Val)]) {
            todo!("e382 adds no virtual assignment")
        }

        fn operand_to_index(&mut self, value: Val) -> RegNode {
            RegNode(i32::try_from(value.0).unwrap_or_default())
        }
    }

    /// `sentient.for %iv = 0 to %bound iter_args(%arg = %init)`, the bound in `bound_locale` and the
    /// carried value in `carried_locale`.
    fn for_op(iv: u32, arg: u32, bound_locale: RegType, carried_locale: RegType, body: Vec<Op>) -> Op {
        Op::Sentient(sentient::Op::For {
            iv: Val(iv),
            bound: Val(90),
            bound_reg: Some(Reg {
                locale: bound_locale,
                index: None,
            }),
            carried: vec![Carried {
                init: Val(91),
                arg: Val(arg),
                result: Val(arg + 100),
                reg: Reg {
                    locale: carried_locale,
                    index: None,
                },
                program_header: false,
                element_size: None,
            }],
            dbg_name: None,
            body,
        })
    }

    /// `sentient.scalar_copy %in` into `locale`.
    fn scalar_copy(result: u32, locale: RegType) -> Op {
        Op::Sentient(sentient::Op::ScalarCopy {
            input: Val(92),
            result: Val(result),
            reg: Reg {
                locale,
                index: None,
            },
            element_size: None,
            program_header: false,
        })
    }

    /// `sentient.scalar_add %a, %b` into `locale`.
    fn scalar_add(result: u32, locale: RegType) -> Op {
        Op::Sentient(sentient::Op::ScalarAdd {
            lhs: Val(93),
            rhs: Val(94),
            result: Val(result),
            reg: Some(Reg {
                locale,
                index: None,
            }),
            element_size: None,
            ty: ScalarTy::Index,
        })
    }

    /// The register `op` carries, by the field each form keeps it in.
    fn assigned(op: &Op) -> Option<RegIndex> {
        match op {
            Op::Sentient(sentient::Op::ScalarCopy { reg, .. }) => reg.index,
            Op::Sentient(sentient::Op::ScalarAdd { reg, .. }) => reg.as_ref().and_then(|r| r.index),
            _ => None,
        }
    }

    /// e382 — the eight locales are coloured in order, a value's register IS its colour, LCCR is the
    /// loop nesting depth the `yield` gives back, and a locale no arm accepts is refused as data with
    /// the op left unassigned.
    #[test]
    fn e382_assigns_each_value_its_colour_and_counts_lccr_by_loop_depth() {
        let mut pass: SmartRegisterAllocation<ColouringGraphs> = SmartRegisterAllocation::default();
        let mut unit = unit_of(vec![
            for_op(
                10,
                11,
                RegType::Lccr,
                RegType::Lrf,
                vec![
                    scalar_copy(30, RegType::Mvr),
                    Op::Sentient(sentient::Op::Yield {
                        results: vec![Val(30)],
                    }),
                ],
            ),
            scalar_add(40, RegType::Ebr),
            for_op(20, 21, RegType::Lccr, RegType::Lrf, Vec::new()),
        ]);
        let mut liveness = NumberedLiveness;

        pass.perform_graph_coloring(&mut unit, &mut liveness, GreedyAllocator::No);

        assert_eq!(pass.reg_graphs.asked, COLOURED_LOCALES.to_vec());
        assert_eq!(pass.reg_graphs.greedy, vec![GreedyAllocator::No; 8]);
        let Op::Sentient(sentient::Op::For {
            bound_reg,
            carried,
            body,
            ..
        }) = &unit.body[0]
        else {
            panic!("the first op is the loop")
        };
        // LCCR0 for the outer loop, and the carried value's own LRF colour.
        assert_eq!(bound_reg.as_ref().and_then(|r| r.index), Some(RegIndex::at::<0>()));
        assert_eq!(carried[0].reg.index, Some(RegIndex::at::<2>()));
        assert_eq!(assigned(&body[0]), Some(RegIndex::at::<5>()));
        // EBR is not in the add's chain: refused as data, and the op keeps no register.
        assert_eq!(assigned(&unit.body[1]), None);
        assert_eq!(pass.failures().len(), 1);
        assert_eq!(pass.failures()[0].locale, Some(RegType::Ebr));
        assert_eq!(
            pass.failures()[0].message,
            "Unknown locale at the add operation"
        );
        // ⭐ THE `yield` GAVE LCCR0 BACK, so the second loop gets it again rather than LCCR1.
        let Op::Sentient(sentient::Op::For { bound_reg, .. }) = &unit.body[2] else {
            panic!("the third op is the second loop")
        };
        assert_eq!(bound_reg.as_ref().and_then(|r| r.index), Some(RegIndex::at::<0>()));
    }

    /// HOW MANY UNITS `FreshLiveness` WAS COMPUTED OVER — e540 constructs its analysis rather than
    /// being handed one, so a static is the only place the count can be read back from.
    static LIVE_RANGES_COMPUTED: AtomicU32 = AtomicU32::new(0);

    /// A `Liveness` THAT RECORDS BEING CONSTRUCTED AND COMPUTED, which is e540's own contribution.
    #[derive(Debug, Clone, Copy, Default)]
    struct FreshLiveness;

    impl Liveness for FreshLiveness {
        fn update_live_ranges_for_program_header_promotion(&mut self, _candidate: Val) {}

        fn is_live_range_overlaps(&self, _val1: Val, _val2: Val) -> bool {
            false
        }

        fn clear(&mut self, _virtual_assigns: VirtualAssigns) {
            todo!("no unit here clears this fake")
        }

        fn compute_register_live_range(&mut self, _unit: &[Op]) {
            LIVE_RANGES_COMPUTED.fetch_add(1, Ordering::Relaxed);
        }

        fn add_virtual_assign_optional(&mut self, _set_of_subsets: &[Vec<Val>]) {
            todo!("no unit here offers a subset to this fake")
        }

        fn add_virtual_assign_enforced(&mut self, _set_of_pairs: &[(Val, Val)]) {
            todo!("no unit here enforces a pair on this fake")
        }

        fn operand_to_index(&mut self, _value: Val) -> RegNode {
            todo!("e540 never reaches a colouring lookup through this fake")
        }
    }

    /// e540 — the entry walks the module's units, and the first one gets a live range computed over
    /// its OWN body before its graphs are built, then reaches the unported colouring.
    #[test]
    fn e540_computes_a_fresh_liveness_per_unit_before_allocating() {
        let mut pass = SmartRegisterAllocation {
            reg_graphs: CountingGraphs::default(),
            register_assignment: Vec::new(),
            failures: Vec::new(),
        };
        let mut program: Program<Dd2, AnyModel, AnyRung> = Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            preamble: Vec::new(),
            units: ProgramUnits::of(
                unit_of(vec![scalar_const(1, 0), scalar_const(2, 1)]),
                vec![unit_of(Vec::new())],
            ),
            bound: core::marker::PhantomData,
        };
        LIVE_RANGES_COMPUTED.store(0, Ordering::Relaxed);

        let reached = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pass.run_on_operation::<Dd2, AnyModel, AnyRung, FreshLiveness>(&mut program);
        }));

        assert!(
            reached.is_err(),
            "the fake has no equivalence classes, so the first unit's colouring panics"
        );
        // The first unit only: it was computed over, cleaned and built over its two ops.
        assert_eq!(LIVE_RANGES_COMPUTED.load(Ordering::Relaxed), 1);
        assert_eq!(
            pass.reg_graphs,
            CountingGraphs {
                cleaned: 1,
                built_over: Some((2, 1)),
            }
        );
    }
}
