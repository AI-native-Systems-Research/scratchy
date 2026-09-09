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

//! `SpecializedCanonicalization.cpp` — 8 of the campaign's 656 units (dependency level(s) [0, 1, 2]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e218_runOn` | 218 | 0 | 36 | `dcc/src/Transform/Sentient/SpecializedCanonicalization.cpp:65` |
//! | `e219_runOn` | 219 | 0 | 35 | `dcc/src/Transform/Sentient/SpecializedCanonicalization.cpp:102` |
//! | `e220_runOn` | 220 | 0 | 31 | `dcc/src/Transform/Sentient/SpecializedCanonicalization.cpp:138` |
//! | `e221_initialize` | 221 | 0 | 20 | `dcc/src/Transform/Sentient/SpecializedCanonicalization.cpp:193` |
//! | `e222_clear` | 222 | 0 | 7 | `dcc/src/Transform/Sentient/SpecializedCanonicalization.cpp:214` |
//! | `e383_runOn` | 383 | 1 | 17 | `dcc/src/Transform/Sentient/SpecializedCanonicalization.cpp:170` |
//! | `e384_runOn` | 384 | 1 | 4 | `dcc/src/Transform/Sentient/SpecializedCanonicalization.cpp:188` |
//! | `e479_runOnOperation` | 479 | 2 | 5 | `dcc/src/Transform/Sentient/SpecializedCanonicalization.cpp:58` |

// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET, so every item below is reachable only from this
// file's own tests until `e479_runOnOperation` lands and something calls it. CI runs clippy with
// `-D warnings`, so without this the first ported leaf of the module fails the gate.
// ⭐ REMOVE THIS WITH `e479_runOnOperation`: at that point an unused item here is a real defect again.
#![allow(dead_code)]

use crate::arch::Arch;
use crate::islands::dataflow_ir::dialects::{dataflow, symbol};
use crate::islands::sentient::Program;
use crate::islands::sentient::dialects::{self, Op, Val};
use crate::model::Model;
use crate::units::{Core, Corelet, DfirUnit, NumFolds, Residency};
use crate::workload::Workload;

/// WHETHER THE OP THE WALK REACHED IS ALREADY WHERE THIS PASS WOULD PUT IT.
///
/// ⭐ THE WHOLE OF `curr_op->getParentOp() == curr_func_ && dom_info.dominates(&*curr_op,
/// insert_point_)`, WHICH IS A BLOCK POSITION AND NOT A GRAPH QUERY. Same-block dominance is "comes
/// earlier", the entry block of a [`Program`] is its `preamble` followed by one
/// `dataflow.program_unit` per unit, and `insert_point_` is the FIRST of those — so an op at function
/// scope dominates it exactly when it sits in the preamble, and no other parent can.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Site {
    /// At function scope and before the first program unit — `dominates` is true.
    Preamble,
    /// Inside a program unit or deeper — either parent or dominance fails.
    Nested,
}

/// `dcc::getCoreletId` (`Utils/DccExtContext.cpp:116`) OFF A [`Residency`]; `None` is its -1.
///
/// ⛔⛔ `CoreWide` IS `corelet = 0`, NOT ABSENT, AND THE KEY TURNS ON IT. A `CoreWide` unit prints
/// `core` AND `corelet = 0` (`UnitMaterializer.cpp:62-80`), so the reference reads back the same pair
/// for it as for `Corelet { corelet: 0 }` and gives the two ONE key. Splitting them would leave a
/// duplicate `dataflow.get_unit` standing.
fn corelet_of(residency: Residency) -> Option<Corelet> {
    match residency {
        Residency::Global | Residency::Scratchpad { .. } => None,
        Residency::CoreWide { .. } => Corelet::checked(0),
        Residency::Corelet { corelet, .. } => Some(corelet),
    }
}

/// THE KEY `dcc::utils::getUnitNameAsString` BUILDS (`Analyses/Utils.cpp:635-641`), AS A KEY.
///
/// ⭐ A TUPLE RATHER THAN ITS STRING, AND EXACTLY AS DISCRIMINATING: the reference concatenates
/// `type.lower()`, `"core"`, the core id, `"corelet"`, the corelet id and `"folds"`, and the
/// alphabetic separators make that concatenation injective — so two keys are equal here exactly when
/// the two strings are. [`DfirUnit::spelling`] is already lower-case and distinct per variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GetUnitKey {
    /// `unit.getType().lower()`.
    unit: DfirUnit,
    /// `dcc::getCoreId(unit)` (`Utils/DccExtContext.cpp:78`).
    core: Option<Core>,
    /// `dcc::getCoreletId(unit)` — see [`corelet_of`].
    corelet: Option<Corelet>,
    /// `unit.getNumResults()`, which is the fold count the op was built with.
    folds: NumFolds,
}

/// THE KEY OFF ONE `dataflow.get_unit`.
fn get_unit_key(residency: Residency, unit: DfirUnit, num_folds: Option<NumFolds>) -> GetUnitKey {
    GetUnitKey {
        unit,
        core: residency.core(),
        corelet: corelet_of(residency),
        folds: num_folds.unwrap_or(NumFolds::ONE),
    }
}

/// TAKE THE OP THAT DEFINES A VALUE OUT OF THE TREE — the first half of `moveBefore`.
///
/// ⛔ REGION COVERAGE IS [`dialects::regions_mut`]'s, the same as
/// [`dialects::erase_defining_op`]'s: a lower-rung region cannot bind a value of this rung.
fn take_defining_op(ops: &mut Vec<Op>, val: Val, taken: &mut Option<Op>) {
    if taken.is_some() {
        return;
    }
    if let Some(at) = ops
        .iter()
        .position(|op| dialects::results(op).contains(&val))
    {
        *taken = Some(ops.remove(at));
        return;
    }
    for op in ops.iter_mut() {
        for region in dialects::regions_mut(op) {
            take_defining_op(region, val, taken);
        }
    }
}

/// `curr_op->moveBefore(insert_point_)` — the op leaves wherever it sits and lands at the END of the
/// preamble, which is the position immediately before the first `dataflow.program_unit`.
fn hoist_into_preamble<A: Arch, M: Model, W: Workload>(program: &mut Program<A, M, W>, val: Val) {
    let mut taken = None;
    take_defining_op(&mut program.preamble, val, &mut taken);
    for unit in program.units.iter_mut() {
        take_defining_op(&mut unit.body, val, &mut taken);
    }
    if let Some(op) = taken {
        program.preamble.push(op);
    }
}

/// `Value::use_empty()` over the whole function.
fn use_empty<A: Arch, M: Model, W: Workload>(program: &Program<A, M, W>, val: Val) -> bool {
    dialects::use_count(val, &program.preamble) == 0
        && program
            .units
            .iter()
            .all(|unit| dialects::use_count(val, &unit.body) == 0)
}

/// `curr_op->replaceAllUsesWith(other)` over the whole function.
fn rewire<A: Arch, M: Model, W: Workload>(program: &mut Program<A, M, W>, of: Val, with: Val) {
    dialects::replace_all_uses_with(&mut program.preamble, of, with);
    for unit in program.units.iter_mut() {
        dialects::replace_all_uses_with(&mut unit.body, of, with);
    }
}

/// `getDefiningOp()` anywhere in the function — the walk's own scope, supplied.
fn defining_op_in_program<'a, A: Arch, M: Model, W: Workload>(
    program: &'a Program<A, M, W>,
    val: Val,
) -> Option<&'a Op> {
    dialects::defining_op(val, &program.preamble).or_else(|| {
        program
            .units
            .iter()
            .find_map(|unit| dialects::defining_op(val, &unit.body))
    })
}

/// THE PASS'S PER-FUNCTION STATE (`:223-228`).
///
/// ⭐ `curr_func_` AND `insert_point_` ARE NOT FIELDS: both exist only to answer "is this op already
/// at function scope, ahead of the first program unit", and that answer is [`Site`]. `opts_` is never
/// read by any of the eight units.
///
/// ⭐ ASSOCIATION LISTS, NOT MAPS: `llvm::StringMap` is keyed by the string these two build, and
/// neither [`GetUnitKey`] nor a [`Core`] is hashable — the lookups are the reference's `find`, and a
/// function's `dataflow.get_unit` count is its unit count.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct SpecializedCanonicalization {
    /// `get_unit_map_`.
    get_unit_map: Vec<(GetUnitKey, Val)>,
    /// `create_symbol_map_`, keyed by `std::to_string(getSymbolID())` — injective, so by the id.
    create_symbol_map: Vec<(i64, Val)>,
    /// `to_be_deleted_`, held by the result of the op to erase.
    to_be_deleted: Vec<Val>,
}

impl SpecializedCanonicalization {
    /// Replaces: e221_initialize
    ///
    /// Opens a function: empty maps, and the insertion point ahead of its first
    /// `dataflow.program_unit`.
    ///
    /// ⛔ THE `return false` ARM IS UNREACHABLE HERE. [`crate::islands::sentient::ProgramUnits`] is
    /// non-empty by construction, so `insert_point_` always exists and the "probably a trace func"
    /// skip has no representation — e383's `if (initialize(func))` is unconditional.
    /// ⛔ AND THE THREE `DT_CHECK`s ARE THE TYPE: this is the only way to make one, so the three
    /// collections are empty by construction rather than by a checked precondition.
    #[must_use]
    pub(crate) fn initialize<A: Arch, M: Model, W: Workload>(
        _program: &Program<A, M, W>,
    ) -> SpecializedCanonicalization {
        SpecializedCanonicalization::default()
    }

    /// Replaces: e222_clear
    ///
    /// Drops everything the pass carried for one function.
    ///
    /// ⛔ UNLIKE `GraphColoring::clear` AND `RegisterGraphs::clean`, THIS ONE RESETS EVERYTHING IT
    /// OWNS — nothing survives into the next function, so it really is `Self::default()`. The
    /// `curr_func_ = nullptr; insert_point_ = nullptr` half is absorbed into [`Site`].
    pub(crate) fn clear(&mut self) {
        *self = SpecializedCanonicalization::default();
    }

    /// The ops `runOn` queued, in the order it queued them — e383 erases these.
    #[must_use]
    pub(crate) fn to_be_deleted(&self) -> &[Val] {
        &self.to_be_deleted
    }

    /// Replaces: e218_runOn
    ///
    /// One `dataflow.get_unit`: queues it if dead, rewires its readers onto the first op with the same
    /// key and queues it if it is a duplicate, else records it and hoists it to the preamble.
    ///
    /// ⛔ THE `IgnoreGetUnitType` GATE (`:67`) CANNOT FIRE: it compares the unit's type against
    /// `-dcc-specialized-canonicalization-ignore-get-unit-type`, a string defaulting to `"-"`
    /// (`:47-50`), and no [`DfirUnit::spelling`] is `-`.
    /// ⛔ TRAP: dead and duplicate ops are QUEUED, not erased — the erase is e383's, after the walk.
    pub(crate) fn run_on_get_unit<A: Arch, M: Model, W: Workload>(
        &mut self,
        curr_op: dataflow::Op,
        site: Site,
        program: &mut Program<A, M, W>,
    ) {
        let dataflow::Op::GetUnit {
            reg_locale: _,
            result,
            residency,
            unit,
            num_folds,
        } = curr_op
        else {
            return;
        };
        if use_empty(program, result) {
            self.to_be_deleted.push(result);
            return;
        }
        let key = get_unit_key(residency, unit, num_folds);
        if let Some(&(_, mapped)) = self.get_unit_map.iter().find(|(seen, _)| *seen == key) {
            rewire(program, result, mapped);
            self.to_be_deleted.push(result);
            return;
        }
        self.get_unit_map.push((key, result));
        if site == Site::Preamble {
            return;
        }
        hoist_into_preamble(program, result);
    }

    /// Replaces: e219_runOn
    ///
    /// One `dataflow.create_group`: hoists it to the preamble, but only once every unit id it reads is
    /// a `dataflow.get_unit` this pass has already mapped.
    ///
    /// ⛔ NO COMMONING, BY THE REFERENCE'S OWN `todo` (`:134-135`) — two identical groups both survive.
    /// ⛔ TRAP: the operand scan is a GUARD, not a fix-up. One unmapped id abandons the whole op, and
    /// the pre-order walk is what normally makes that unreachable.
    pub(crate) fn run_on_create_group<A: Arch, M: Model, W: Workload>(
        &mut self,
        curr_op: dataflow::Op,
        site: Site,
        program: &mut Program<A, M, W>,
    ) {
        let dataflow::Op::CreateGroup { result, unit_ids } = curr_op else {
            return;
        };
        if site == Site::Preamble {
            return;
        }
        for unit_id in unit_ids {
            // `isa<BlockArgument>(unit) || !isa<GetUnitOp>(unit.getDefiningOp())`.
            let Some(Op::Dataflow(dataflow::Op::GetUnit {
                residency,
                unit,
                num_folds,
                ..
            })) = defining_op_in_program(program, unit_id)
            else {
                return;
            };
            let key = get_unit_key(*residency, *unit, *num_folds);
            if !self.get_unit_map.iter().any(|(seen, _)| *seen == key) {
                return;
            }
        }
        hoist_into_preamble(program, result);
    }

    /// Replaces: e220_runOn
    ///
    /// One `symbol.create_symbol`, by symbol id: queues it if dead, rewires and queues it if a symbol
    /// with that id was already seen, else records it and hoists it to the preamble.
    ///
    /// ⛔ TRAP: `maxValue` IS NOT PART OF THE KEY. The reference keys on `getSymbolID()` alone, so the
    /// survivor's bound is the one every reader inherits even where the duplicate carried another.
    pub(crate) fn run_on_create_symbol<A: Arch, M: Model, W: Workload>(
        &mut self,
        curr_op: symbol::Op,
        site: Site,
        program: &mut Program<A, M, W>,
    ) {
        let symbol::Op::CreateSymbol {
            result, symbol_id, ..
        } = curr_op
        else {
            return;
        };
        if use_empty(program, result) {
            self.to_be_deleted.push(result);
            return;
        }
        if let Some(&(_, mapped)) = self
            .create_symbol_map
            .iter()
            .find(|(seen, _)| *seen == symbol_id)
        {
            rewire(program, result, mapped);
            self.to_be_deleted.push(result);
            return;
        }
        self.create_symbol_map.push((symbol_id, result));
        if site == Site::Preamble {
            return;
        }
        hoist_into_preamble(program, result);
    }
}

// crustify:todo: e383_runOn
//   authority : dcc/src/Transform/Sentient/SpecializedCanonicalization.cpp:170  (17 body lines, level 1)
//   original  : void runOn(func::FuncOp func)
//   calls     : e218_runOn, e219_runOn, e220_runOn, e221_initialize, e222_clear, e384_runOn

// crustify:todo: e384_runOn
//   authority : dcc/src/Transform/Sentient/SpecializedCanonicalization.cpp:188  (4 body lines, level 1)
//   original  : void runOn(ModuleOp module_op)
//   calls     : e218_runOn, e219_runOn, e220_runOn, e383_runOn

// crustify:todo: e479_runOnOperation
//   authority : dcc/src/Transform/Sentient/SpecializedCanonicalization.cpp:58  (5 body lines, level 2)
//   original  : void runOnOperation()
//   calls     : e218_runOn, e219_runOn, e220_runOn, e383_runOn, e384_runOn

#[cfg(test)]
mod unit_tests {
    use super::{Site, SpecializedCanonicalization};
    use crate::arch::Dd2;
    use crate::formats::Bits;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::dialects::{dataflow, symbol};
    use crate::islands::dataflow_ir::{GroupId, OpIndex, ProgramName, Units};
    use crate::islands::sentient::dialects::sentient::{Reg, RegType};
    use crate::islands::sentient::dialects::{Op, Val, sentient};
    use crate::islands::sentient::{Program, ProgramUnit, ProgramUnits};
    use crate::model::Model;
    use crate::units::{Core, Corelet, DfirUnit, Residency};
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

    /// A one-unit program: `preamble` at function scope, `body` inside the program unit.
    fn program(preamble: Vec<Op>, body: Vec<Op>) -> Program<Dd2, AnyModel, AnyRung> {
        Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            preamble,
            units: ProgramUnits::of(
                ProgramUnit {
                    iter_arg: None,
                    on: Units::one(DfirUnit::Lxlu, Val(0)),
                    precision: None,
                    body,
                    arch: core::marker::PhantomData,
                },
                Vec::new(),
            ),
            bound: core::marker::PhantomData,
        }
    }

    /// Core 0 — the only core these fixtures name.
    fn core0() -> Option<Core> {
        Core::checked(0)
    }

    /// `%r = dataflow.get_unit`.
    fn get_unit(result: u32, residency: Residency, unit: DfirUnit) -> dataflow::Op {
        dataflow::Op::GetUnit {
            reg_locale: None,
            result: Val(result),
            residency,
            unit,
            num_folds: None,
        }
    }

    /// `%r = dataflow.create_group (..)`.
    fn create_group(result: u32, unit_ids: &[u32]) -> dataflow::Op {
        dataflow::Op::CreateGroup {
            result: Val(result),
            unit_ids: unit_ids.iter().map(|&id| Val(id)).collect(),
        }
    }

    /// `%r = symbol.create_symbol {SymbolId = N : i32}`.
    fn create_symbol(result: u32, symbol_id: i64) -> symbol::Op {
        symbol::Op::CreateSymbol {
            result: Val(result),
            symbol_id,
            max_value: None,
        }
    }

    /// `%r = sentient.scalar_copy %in` — a reader, so the op it reads is not dead.
    fn reader(input: u32, result: u32) -> Op {
        Op::Sentient(sentient::Op::ScalarCopy {
            input: Val(input),
            result: Val(result),
            reg: Reg {
                locale: RegType::Lar,
                index: None,
            },
            program_header: false,
            element_size: Some(Bits(32)),
        })
    }

    /// e218 — the vendor's duplicate case, AND the `CoreWide`/`Corelet { corelet: 0 }` collision that
    /// makes these two ops ONE key: the first is hoisted, the second rewired onto it and queued.
    #[test]
    fn e218_a_core_wide_and_a_corelet_zero_get_unit_share_one_key() {
        let core = core0().expect("Dd2 has a core 0");
        let wide = get_unit(1, Residency::CoreWide { core }, DfirUnit::L3lu);
        let corelet_zero = get_unit(
            2,
            Residency::Corelet {
                core,
                corelet: Corelet::checked(0).expect("Dd2 has a corelet 0"),
            },
            DfirUnit::L3lu,
        );
        let mut prog = program(
            Vec::new(),
            vec![
                Op::Dataflow(wide.clone()),
                Op::Dataflow(corelet_zero.clone()),
                Op::Dataflow(create_group(3, &[1, 2])),
            ],
        );
        let mut pass = SpecializedCanonicalization::initialize(&prog);

        pass.run_on_get_unit(wide, Site::Nested, &mut prog);
        pass.run_on_get_unit(corelet_zero, Site::Nested, &mut prog);

        assert_eq!(prog.preamble.len(), 1);
        assert_eq!(
            prog.units.iter().next().expect("one unit").body,
            vec![
                Op::Dataflow(get_unit(
                    2,
                    Residency::Corelet {
                        core,
                        corelet: Corelet::checked(0).expect("a corelet 0"),
                    },
                    DfirUnit::L3lu
                )),
                Op::Dataflow(create_group(3, &[1, 1])),
            ]
        );
        assert_eq!(pass.to_be_deleted(), [Val(2)]);
    }

    /// e218 — a `dataflow.get_unit` nothing reads is queued for deletion and never hoisted.
    #[test]
    fn e218_a_dead_get_unit_is_queued_and_not_hoisted() {
        let dead = get_unit(1, Residency::Global, DfirUnit::Hbm);
        let mut prog = program(Vec::new(), vec![Op::Dataflow(dead.clone())]);
        let mut pass = SpecializedCanonicalization::initialize(&prog);

        pass.run_on_get_unit(dead.clone(), Site::Nested, &mut prog);

        assert_eq!(prog.preamble, Vec::new());
        assert_eq!(
            prog.units.iter().next().expect("one unit").body,
            vec![Op::Dataflow(dead)]
        );
        assert_eq!(pass.to_be_deleted(), [Val(1)]);
    }

    /// e219 — a group all of whose ids are mapped `get_unit`s is hoisted; one whose id is anything
    /// else is abandoned where it stands.
    #[test]
    fn e219_a_group_moves_only_once_every_unit_id_is_a_mapped_get_unit() {
        let unit = get_unit(
            1,
            Residency::Scratchpad {
                core: core0().expect("a core 0"),
            },
            DfirUnit::Lx,
        );
        let group = create_group(2, &[1]);
        let mut prog = program(
            vec![Op::Dataflow(unit.clone())],
            vec![Op::Dataflow(group.clone())],
        );
        let mut pass = SpecializedCanonicalization::initialize(&prog);
        pass.run_on_get_unit(unit.clone(), Site::Preamble, &mut prog);

        pass.run_on_create_group(group.clone(), Site::Nested, &mut prog);

        assert_eq!(
            prog.preamble,
            vec![Op::Dataflow(unit.clone()), Op::Dataflow(group)]
        );
        assert_eq!(prog.units.iter().next().expect("one unit").body, Vec::new());

        // The negative: `%9` is a `sentient.scalar_copy` result, not a `get_unit`.
        let stray = create_group(3, &[9]);
        let mut other = program(
            vec![Op::Dataflow(unit)],
            vec![reader(8, 9), Op::Dataflow(stray.clone())],
        );
        let before = other.clone();
        pass.run_on_create_group(stray, Site::Nested, &mut other);
        assert_eq!(other, before);
    }

    /// e220 — two `symbol.create_symbol`s with one id: the first is hoisted, the second's reader is
    /// rewired onto it and the op is queued.
    #[test]
    fn e220_two_symbols_with_one_id_leave_one_standing() {
        let first = create_symbol(1, 7);
        let second = create_symbol(2, 7);
        let mut prog = program(
            Vec::new(),
            vec![
                Op::Symbol(first.clone()),
                reader(1, 10),
                Op::Symbol(second.clone()),
                reader(2, 11),
            ],
        );
        let mut pass = SpecializedCanonicalization::initialize(&prog);

        pass.run_on_create_symbol(first.clone(), Site::Nested, &mut prog);
        pass.run_on_create_symbol(second.clone(), Site::Nested, &mut prog);

        assert_eq!(prog.preamble, vec![Op::Symbol(first)]);
        assert_eq!(
            prog.units.iter().next().expect("one unit").body,
            vec![reader(1, 10), Op::Symbol(second), reader(1, 11)]
        );
        assert_eq!(pass.to_be_deleted(), [Val(2)]);
    }

    /// e221 — a program always has its insert point, so opening one is unconditional and hands back
    /// three empty collections.
    #[test]
    fn e221_opens_a_function_with_nothing_carried_over() {
        let prog = program(Vec::new(), Vec::new());

        let pass = SpecializedCanonicalization::initialize(&prog);

        assert_eq!(pass, SpecializedCanonicalization::default());
        assert_eq!(pass.to_be_deleted(), [] as [Val; 0]);
    }

    /// e222 — everything the pass carried for a function goes, unlike its two `Analyses/` neighbours.
    #[test]
    fn e222_drops_every_collection() {
        let dead = get_unit(1, Residency::Global, DfirUnit::Hbm);
        let mut prog = program(Vec::new(), vec![Op::Dataflow(dead.clone())]);
        let mut pass = SpecializedCanonicalization::initialize(&prog);
        pass.run_on_get_unit(dead, Site::Nested, &mut prog);
        assert_eq!(pass.to_be_deleted(), [Val(1)]);

        pass.clear();

        assert_eq!(pass, SpecializedCanonicalization::default());
    }
}
