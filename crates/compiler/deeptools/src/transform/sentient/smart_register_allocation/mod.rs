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

use super::analyses::{Liveness, RegisterGraphs};
use crate::arch::Arch;
use crate::islands::sentient::dialects::sentient::RegType;
use crate::islands::sentient::dialects::{Definitions, Op, Val, sentient};
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

// crustify:todo: e382_performGraphColoring
//   authority : dcc/src/Transform/Sentient/SmartRegisterAllocation.cpp:136  (484 body lines, level 1)
//   original  : void SmartRegisterAllocationPass::performGraphColoring( dataflow::ProgramUnitOp &unit, Liveness &liveness, bool use_greedy_allocator)
//   calls     : e252_size

/// `useGreedyAllocator = (opts_.OptLevel == 0)`, set once in the pass constructor (`:48`).
///
/// ⛔ NOT A `dcc-opt` FLAG BUT A BUILD OPTION: `CommonPassOptions::OptLevel` defaults to `-1`
/// (`dcc/tools/Options/dcc-pass-option.h:117`), so an ordinary build colours normally and only `-O0`
/// is greedy. The `Passes.td` option of the same name (`Passes.td:365`) is overwritten by that
/// constructor line and never read.
const USE_GREEDY_ALLOCATOR: bool = false;

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

    /// `performGraphColoring(unit, liveness, use_greedy_allocator)` — entry 382, level 1, not yet
    /// ported.
    fn perform_graph_coloring<A: Arch, L: Liveness>(
        &mut self,
        unit: &mut ProgramUnit<A>,
        liveness: &mut L,
        use_greedy_allocator: bool,
    ) -> ! {
        let _ = (unit, liveness, use_greedy_allocator);
        todo!(
            "e382_performGraphColoring(unit, liveness, use_greedy_allocator) — the per-locale \
             hyper-graph colouring and the register each value is assigned \
             (SmartRegisterAllocation.cpp:136), 484 lines"
        )
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
    /// ⚠️ `DEBUG_WITH_TYPE(VerboseDebug, liveness.dump())` IS DROPPED (`:631`): it changes no IR.
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
    use super::{Liveness, RegisterGraphs, SmartRegisterAllocation, is_known_to_have_same_values};
    use crate::arch::{Dd2, Elements};
    use crate::formats::Bits;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::link::SendEnd;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::dataflow_ir::{GroupId, OpIndex, ProgramName, Units};
    use crate::islands::sentient::dialects::sentient::{Reg, RegType, ShuffleMode};
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
            todo!("no unit here folds equivalence classes through this fake")
        }

        fn build_hyper_graph(&mut self, _locale: RegType) {
            todo!("no unit here builds a hyper-graph through this fake")
        }

        fn fast_check_colorability(&mut self, _num_colors: MaxRegNum, _locale: RegType) -> bool {
            todo!("no unit here checks colorability through this fake")
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
        };
        let mut unit = unit_of(vec![scalar_const(1, 0), scalar_const(2, 1)]);
        let mut liveness = SilentLiveness;

        let reached = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pass.do_register_allocation(&mut unit, &mut liveness);
        }));

        assert!(
            reached.is_err(),
            "e382 is not ported, so the colouring panics"
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
    }

    /// e540 — the entry walks the module's units, and the first one gets a live range computed over
    /// its OWN body before its graphs are built, then reaches the unported colouring.
    #[test]
    fn e540_computes_a_fresh_liveness_per_unit_before_allocating() {
        let mut pass = SmartRegisterAllocation {
            reg_graphs: CountingGraphs::default(),
            register_assignment: Vec::new(),
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
            "e382 is not ported, so the first unit's colouring panics"
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
