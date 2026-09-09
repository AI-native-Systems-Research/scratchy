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

//! `RegisterPacking.cpp` — 13 of the campaign's 656 units (dependency level(s) [0, 1, 2, 3, 4]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e132_setNewRegisterIndex` | 132 | 0 | 4 | `dcc/src/Transform/Sentient/RegisterPacking.cpp:75` |
//! | `e133_cleanup` | 133 | 0 | 1 | `dcc/src/Transform/Sentient/RegisterPacking.cpp:135` |
//! | `e134_doRenumbering` | 134 | 0 | 7 | `dcc/src/Transform/Sentient/RegisterPacking.cpp:150` |
//! | `e135_getRegTypeAndIndex` | 135 | 0 | 22 | `dcc/src/Transform/Sentient/RegisterPacking.cpp:249` |
//! | `e136_fillTypedRegCollection` | 136 | 0 | 31 | `dcc/src/Transform/Sentient/RegisterPacking.cpp:298` |
//! | `e348_updateRegIndex` | 348 | 1 | 16 | `dcc/src/Transform/Sentient/RegisterPacking.cpp:162` |
//! | `e349_getRegWithType` | 349 | 1 | 7 | `dcc/src/Transform/Sentient/RegisterPacking.cpp:239` |
//! | `e459_computeNewRegisterIndices` | 459 | 2 | 53 | `dcc/src/Transform/Sentient/RegisterPacking.cpp:182` |
//! | `e460_runOnAllOps` | 460 | 2 | 21 | `dcc/src/Transform/Sentient/RegisterPacking.cpp:275` |
//! | `e461_runOnForOp` | 461 | 2 | 38 | `dcc/src/Transform/Sentient/RegisterPacking.cpp:333` |
//! | `e462_runOnCopyOp` | 462 | 2 | 12 | `dcc/src/Transform/Sentient/RegisterPacking.cpp:373` |
//! | `e522_runOnProgramUnitOp` | 522 | 3 | 21 | `dcc/src/Transform/Sentient/RegisterPacking.cpp:388` |
//! | `e573_runOnOperation` | 573 | 4 | 11 | `dcc/src/Transform/Sentient/RegisterPacking.cpp:410` |

#![allow(dead_code)]
// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET — `e573_runOnOperation` (level 4) is what calls this
// file's driver, and every unit below is reachable only from the tests until it lands. CI runs clippy
// with `-D warnings`. ⭐ REMOVE THIS WITH e573.

use crate::islands::sentient::dialects::sentient as ops;
use crate::islands::sentient::dialects::{
    Definitions, Op, Val, set_value_reg_index, symbol, uniform,
};

/// ONE REGISTER-RELATED SSA VALUE THE PACKER TRACKS — the pass-local `RegIndex`
/// (`RegisterPacking.cpp:60-92`), renamed because [`ops::RegIndex`] is the island's index type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrackedReg {
    /// `val_`.
    pub value: Val,
    /// `old_register_index_` — the index the op carried on entry; `None` is the `.td`'s `-1`.
    pub old_index: Option<ops::RegIndex>,
    /// `new_register_index_` — the packed index, `None` until one is chosen (the reference's `-1`).
    pub new_index: Option<ops::RegIndex>,
    /// `is_index_updated_`.
    pub index_updated: bool,
    /// `has_same_value_in_all_units_`.
    pub same_value_in_all_units: bool,
    /// `is_header_promoted_`.
    pub header_promoted: bool,
}

impl TrackedReg {
    /// `RegIndex(val, reg_index)` (`RegisterPacking.cpp:62`) — the two-argument constructor.
    #[must_use]
    pub const fn new(value: Val, old_index: Option<ops::RegIndex>) -> TrackedReg {
        TrackedReg {
            value,
            old_index,
            new_index: None,
            index_updated: false,
            same_value_in_all_units: false,
            header_promoted: false,
        }
    }

    /// `RegIndex(val, reg_index, is_header_promoted, has_same_value_in_all_units)`
    /// (`RegisterPacking.cpp:64`) — every caller of it passes `is_header_promoted = true`.
    #[must_use]
    pub const fn promoted(
        value: Val,
        old_index: Option<ops::RegIndex>,
        same_value_in_all_units: bool,
    ) -> TrackedReg {
        TrackedReg {
            value,
            old_index,
            new_index: None,
            index_updated: false,
            same_value_in_all_units,
            header_promoted: true,
        }
    }
}

/// THE REGISTERS OF ONE REGISTER FILE — `TypedRegCollection` (`RegisterPacking.cpp:94-107`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedRegCollection {
    /// `reg_type_`.
    pub reg_type: ops::RegType,
    /// `regs_list_`.
    pub regs: Vec<TrackedReg>,
}

impl TypedRegCollection {
    /// `TypedRegCollection(reg_type)` (`RegisterPacking.cpp:97`).
    #[must_use]
    pub const fn new(reg_type: ops::RegType) -> TypedRegCollection {
        TypedRegCollection {
            reg_type,
            regs: Vec::new(),
        }
    }
}

/// `RegisterPackingPass`'s own state — the register table it fills, packs and writes back.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RegisterPacking {
    /// `reg_table`.
    pub reg_table: Vec<TypedRegCollection>,
}

impl TrackedReg {
    /// Replaces: e132_setNewRegisterIndex
    ///
    /// Records the packed index for this register and marks it renumbered.
    ///
    /// ⚠️ TRAP: `is_index_updated_` GOES TRUE EVEN FOR `-1`. The reference sets the flag
    /// unconditionally (`RegisterPacking.cpp:75-78`), so a register handed an unassigned index still
    /// counts as decided by `updateRegIndex` (e348) and by [`RegisterPacking::do_renumbering`].
    pub fn set_new_register_index(&mut self, index: Option<ops::RegIndex>) {
        self.new_index = index;
        self.index_updated = true;
    }
}

impl RegisterPacking {
    /// Replaces: e133_cleanup
    ///
    /// Drops the register table — the pass runs this per `dataflow.program_unit`, before deciding
    /// whether that unit is even packed (`RegisterPacking.cpp:416-417`).
    pub fn cleanup(&mut self) {
        self.reg_table.clear();
    }
}

impl RegisterPacking {
    /// Replaces: e134_doRenumbering
    ///
    /// Writes every tracked register's packed index back onto the value that holds it.
    ///
    /// ⚠️ TRAP: `is_index_updated_` IS NOT CONSULTED. A register the packer never renumbered has
    /// `new_register_index_ == -1` and that `-1` is written back over the index it came in with
    /// (`RegisterPacking.cpp:150-156`).
    /// ⚠️ TRAP: an index is only ever COPIED here, never minted. Choosing a new one is e348/e459's
    /// problem, and [`ops::RegIndex::allocated`] is the one door for it.
    pub fn do_renumbering(&self, unit: &mut [Op]) {
        for collection in &self.reg_table {
            for reg in &collection.regs {
                set_value_reg_index(unit, reg.value, reg.new_index);
            }
        }
    }
}

/// Replaces: e135_getRegTypeAndIndex
///
/// The register file and index an op holds at `position`, or `None` when it holds none there.
///
/// ⭐ THE SIX `DT_CHECK`s (`:252`, `:255`, `:258`; `:262`, `:265`, `:268`) ARE DISCHARGED BY THE
/// TYPE: an island [`ops::Reg`] is always a locale AND an index, so "has `regIndex` but no
/// `regLocale`" is not constructible.
/// ⭐ `position` IS IGNORED FOR AN OP WITH THE SINGULAR `regIndex`, exactly as the reference ignores
/// its `idx` on that branch (`RegisterPacking.cpp:251-261`).
#[must_use]
pub fn reg_type_and_index(op: &Op, position: usize) -> Option<ops::Reg> {
    let Op::Sentient(inner) = op else {
        return None;
    };
    match inner {
        ops::Op::LoadAndSend { reg, .. }
        | ops::Op::ReceiveAndStore { reg, .. }
        | ops::Op::LoadComputeAndSend { reg, .. }
        | ops::Op::ScalarCopy { reg, .. }
        | ops::Op::ReceiveAndExtractScalar { reg, .. } => Some(*reg),
        ops::Op::ScalarAdd { reg, .. } | ops::Op::ScalarSub { reg, .. } => *reg,
        ops::Op::LoadAndStore {
            src_reg, dst_reg, ..
        } => match position {
            0 => Some(*src_reg),
            1 => Some(*dst_reg),
            _ => None,
        },
        ops::Op::LoadAndExtractScalar {
            addr_reg, data_reg, ..
        } => match position {
            0 => Some(*addr_reg),
            1 => Some(*data_reg),
            _ => None,
        },
        // ⭐ THE REFERENCE'S `1 + 2n` NUMBERING, WHICH ITS OWN CALLERS COMPUTE: entry 0 is the
        // induction variable, `1..=n` the region arguments and the rest the results
        // (`RegisterPacking.cpp:337`, `:342`, `:357-358`). ⛔ POSITION 0 HAS NO FIELD IN THIS
        // ISLAND — a `Carried` is one `Reg` per carried value and none for the bound; see
        // [`ops::Carried`].
        ops::Op::For { carried, .. } => {
            let carried_count = carried.len();
            position
                .checked_sub(1)
                .and_then(|p| carried.get(p).or_else(|| carried.get(p - carried_count)))
                .map(|value| value.reg)
        }
        ops::Op::If { yielded, .. } => yielded.get(position).map(|value| value.reg),
        // ⭐ NO REGISTER ARRAYS AT ALL — the reference reaches these through neither branch, because
        // `runOnProgramUnitOp` only visits an op that HAS `regIndex` or `regIndices` (`:396`).
        _ => None,
    }
}

/// Replaces: e136_fillTypedRegCollection
///
/// Adds one tracked register to `collection`, deciding for a header-promoted value whether it holds
/// the same thing in every unit.
///
/// ⚠️ TRAP: A PROMOTED VALUE DEFINED BY ANYTHING ELSE IS SILENTLY DROPPED. The reference's chain has
/// no `else` (`RegisterPacking.cpp:301-325`), so a promoted register initialised by anything other
/// than `sentient.scalar_constant`, `symbol.create_symbol` or `uniform.query_map` is added to NO
/// collection and the packer never sees it.
pub fn fill_typed_reg_collection(
    collection: &mut TypedRegCollection,
    init_value: Val,
    res_value: Val,
    reg_index: Option<ops::RegIndex>,
    header_promoted: bool,
    defs: Definitions<'_>,
) {
    if !header_promoted {
        collection.regs.push(TrackedReg::new(res_value, reg_index));
        return;
    }
    match defs.of(init_value) {
        Some(
            Op::Sentient(ops::Op::ScalarConstant { .. })
            | Op::Symbol(symbol::Op::CreateSymbol { .. }),
        ) => {
            collection
                .regs
                .push(TrackedReg::promoted(res_value, reg_index, true));
        }
        Some(Op::Uniform(uniform::Op::QueryMap { map, .. })) => {
            let values = constant_target_values(*map, defs);
            // ⭐ `llvm::all_of` OVER AN EMPTY RANGE IS TRUE AND NEVER CALLS `.front()`, and empty is
            // exactly what `getConstantTargetValues` returns when a target is not a constant — so
            // "not all constants" reaches the same arm as "all equal".
            let same = values
                .first()
                .is_none_or(|first| values.iter().all(|v| v == first));
            collection
                .regs
                .push(TrackedReg::promoted(res_value, reg_index, same));
        }
        _ => {}
    }
}

/// `dcc::uniform::utils::getConstantTargetValues` (`Dialect/Uniform/Utils.cpp:382`) — a query map's
/// per-unit target constants, EMPTY when any one of them is not a `sentient.scalar_constant`.
///
/// ⛔ NOT AN ANCHORED UNIT — `Dialect/Uniform/Utils.cpp` is outside this campaign's file list. The
/// reference dereferences the `getDefiningOp<DefImmutableMappingOp>()` without a check and
/// `DT_CHECK`s the mapping non-empty, so both are a crash there and a named `todo!` here.
pub(crate) fn constant_target_values(map: Val, defs: Definitions<'_>) -> Vec<i64> {
    let pairs = match defs.of(map) {
        Some(Op::Uniform(uniform::Op::DefImmutableMapping { pairs, .. })) => pairs.clone(),
        other => todo!(
            "a uniform.query_map's $map is not a uniform.def_immutable_mapping \
             (Dialect/Uniform/Utils.cpp:385): {other:?}"
        ),
    };
    if pairs.is_empty() {
        todo!(
            "getConstantTargetValues: DT_CHECK(!immutable_map.getValues().empty()) \
             (Dialect/Uniform/Utils.cpp:387)"
        );
    }
    let mut values: Vec<i64> = Vec::new();
    for (_key, value) in pairs {
        match defs.of(value) {
            Some(Op::Sentient(ops::Op::ScalarConstant { value, .. })) => values.push(*value),
            _ => return Vec::new(),
        }
    }
    values
}

impl TypedRegCollection {
    /// Replaces: e348_updateRegIndex
    ///
    /// Gives the register at `at` its packed index — its own `-1` back, else the index an equally
    /// numbered register already got, else `suggested` — and answers whether `suggested` was consumed.
    ///
    /// ⛔ THE REGISTER IS A POSITION IN THIS COLLECTION, NOT A SECOND BORROW. The reference's `reg` is
    /// an element of the very `getRegistersList()` it then scans (`RegisterPacking.cpp:169-175`, and
    /// e459 calls it with exactly that aliasing pair), which two `&mut` cannot say.
    /// ⭐ `suggested_index` IS AN INDEX AND NEVER `-1`: e459's `idx` starts at 0 and only rises, so the
    /// type is [`ops::RegIndex`] rather than an `Option` of one.
    pub fn update_reg_index(&mut self, at: usize, suggested: ops::RegIndex) -> bool {
        let old = self.regs[at].old_index;
        if old.is_none() {
            self.regs[at].set_new_register_index(None);
            return false;
        }
        let inherited = self
            .regs
            .iter()
            .find(|other| other.old_index == old && other.index_updated)
            .map(|other| other.new_index);
        match inherited {
            Some(new_index) => {
                self.regs[at].set_new_register_index(new_index);
                false
            }
            None => {
                self.regs[at].set_new_register_index(Some(suggested));
                true
            }
        }
    }
}

/// Replaces: e349_getRegWithType
///
/// Where `reg_type`'s collection sits in the table, appending an empty one when the table has none.
///
/// ⭐ A POSITION, NOT THE `TypedRegCollection *` THE REFERENCE RETURNS: the caller mutates the table
/// through it and the reference's pointer is invalidated by its own `push_back`.
pub fn reg_with_type(reg_table: &mut Vec<TypedRegCollection>, reg_type: ops::RegType) -> usize {
    match reg_table
        .iter()
        .position(|collection| collection.reg_type == reg_type)
    {
        Some(at) => at,
        None => {
            reg_table.push(TypedRegCollection::new(reg_type));
            reg_table.len() - 1
        }
    }
}

// crustify:todo: e459_computeNewRegisterIndices
//   authority : dcc/src/Transform/Sentient/RegisterPacking.cpp:182  (53 body lines, level 2)
//   original  : void RegisterPackingPass::computeNewRegisterIndices( std::map<SentientRegType, std::vector<int>> &avoid_renumbering_regs)
//   calls     : e132_setNewRegisterIndex, e252_size, e348_updateRegIndex

// crustify:todo: e460_runOnAllOps
//   authority : dcc/src/Transform/Sentient/RegisterPacking.cpp:275  (21 body lines, level 2)
//   original  : void RegisterPackingPass::runOnAllOps(mlir::Operation *op)
//   calls     : e135_getRegTypeAndIndex, e349_getRegWithType

// crustify:todo: e461_runOnForOp
//   authority : dcc/src/Transform/Sentient/RegisterPacking.cpp:333  (38 body lines, level 2)
//   original  : void RegisterPackingPass::runOnForOp(sentient::ForOp for_op)
//   calls     : e135_getRegTypeAndIndex, e136_fillTypedRegCollection, e252_size, e349_getRegWithType

// crustify:todo: e462_runOnCopyOp
//   authority : dcc/src/Transform/Sentient/RegisterPacking.cpp:373  (12 body lines, level 2)
//   original  : void RegisterPackingPass::runOnCopyOp(sentient::CopyOp copy_op)
//   calls     : e136_fillTypedRegCollection, e349_getRegWithType

// crustify:todo: e522_runOnProgramUnitOp
//   authority : dcc/src/Transform/Sentient/RegisterPacking.cpp:388  (21 body lines, level 3)
//   original  : void RegisterPackingPass::runOnProgramUnitOp(dataflow::ProgramUnitOp unit)
//   calls     : e134_doRenumbering, e459_computeNewRegisterIndices, e460_runOnAllOps, e461_runOnForOp, e462_runOnCopyOp

// crustify:todo: e573_runOnOperation
//   authority : dcc/src/Transform/Sentient/RegisterPacking.cpp:410  (11 body lines, level 4)
//   original  : void RegisterPackingPass::runOnOperation()
//   calls     : e133_cleanup, e522_runOnProgramUnitOp

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::islands::dataflow_ir::ty::ScalarTy;

    /// `sentient.scalar_constant` — an `init_value` the promoted branch recognises.
    fn scalar_constant(result: Val, value: i64) -> Op {
        Op::Sentient(ops::Op::ScalarConstant {
            value,
            result,
            reg_locale: ops::RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    /// `sentient.scalar_copy` — one op with the singular `regLocale`/`regIndex` pair.
    fn scalar_copy(result: Val, locale: ops::RegType, index: Option<ops::RegIndex>) -> Op {
        Op::Sentient(ops::Op::ScalarCopy {
            input: Val(0),
            result,
            reg: ops::Reg { locale, index },
            element_size: None,
            program_header: false,
        })
    }

    /// The `reg` a `sentient.scalar_copy` in `ops_list` now carries.
    fn copy_reg(ops_list: &[Op]) -> ops::Reg {
        match &ops_list[0] {
            Op::Sentient(ops::Op::ScalarCopy { reg, .. }) => *reg,
            other => panic!("not a scalar_copy: {other:?}"),
        }
    }

    /// `sentient.for` carrying two values, each with its own register.
    fn for_op(first: ops::RegType, second: ops::RegType) -> Op {
        let carried = |init, arg, result, locale| ops::Carried {
            init,
            arg,
            result,
            reg: ops::Reg {
                locale,
                index: None,
            },
            program_header: false,
            element_size: None,
        };
        Op::Sentient(ops::Op::For {
            iv: Val(1),
            bound: Val(2),
            bound_reg: None,
            carried: vec![
                carried(Val(3), Val(4), Val(5), first),
                carried(Val(6), Val(7), Val(8), second),
            ],
            dbg_name: None,
            body: Vec::new(),
        })
    }

    /// e132_setNewRegisterIndex — the index lands, and the flag goes up even for an unassigned one.
    #[test]
    fn e132_set_new_register_index() {
        let mut reg = TrackedReg::new(Val(9), Some(ops::RegIndex::at::<5>()));
        reg.set_new_register_index(Some(ops::RegIndex::at::<2>()));
        assert_eq!(reg.new_index, Some(ops::RegIndex::at::<2>()));
        assert!(reg.index_updated);

        let mut unassigned = TrackedReg::new(Val(9), None);
        unassigned.set_new_register_index(None);
        assert_eq!(unassigned.new_index, None);
        assert!(unassigned.index_updated);
    }

    /// e133_cleanup — the table is empty again.
    #[test]
    fn e133_cleanup() {
        let mut packing = RegisterPacking {
            reg_table: vec![TypedRegCollection::new(ops::RegType::Lrf)],
        };
        packing.cleanup();
        assert!(packing.reg_table.is_empty());
    }

    /// e134_doRenumbering — the packed index reaches the op, and an unrenumbered register writes
    /// its `-1` over the index the op came in with.
    #[test]
    fn e134_do_renumbering() {
        let mut unit = vec![scalar_copy(
            Val(9),
            ops::RegType::Lrf,
            Some(ops::RegIndex::at::<7>()),
        )];
        let mut reg = TrackedReg::new(Val(9), Some(ops::RegIndex::at::<7>()));
        reg.set_new_register_index(Some(ops::RegIndex::at::<1>()));
        let packing = RegisterPacking {
            reg_table: vec![TypedRegCollection {
                reg_type: ops::RegType::Lrf,
                regs: vec![reg],
            }],
        };
        packing.do_renumbering(&mut unit);
        assert_eq!(copy_reg(&unit).index, Some(ops::RegIndex::at::<1>()));

        let never_packed = RegisterPacking {
            reg_table: vec![TypedRegCollection {
                reg_type: ops::RegType::Lrf,
                regs: vec![TrackedReg::new(Val(9), Some(ops::RegIndex::at::<7>()))],
            }],
        };
        never_packed.do_renumbering(&mut unit);
        assert_eq!(copy_reg(&unit).index, None);
    }

    /// e135_getRegTypeAndIndex — the singular pair ignores the position, and a `sentient.for`'s
    /// `1 + 2n` numbering reaches the region argument and the result of the same carried value.
    #[test]
    fn e135_reg_type_and_index() {
        let copy = scalar_copy(Val(9), ops::RegType::Lrf, Some(ops::RegIndex::at::<4>()));
        let expected = ops::Reg {
            locale: ops::RegType::Lrf,
            index: Some(ops::RegIndex::at::<4>()),
        };
        assert_eq!(reg_type_and_index(&copy, 0), Some(expected));
        assert_eq!(reg_type_and_index(&copy, 3), Some(expected));

        let loop_op = for_op(ops::RegType::Lar, ops::RegType::Lbr);
        let locale = |position| reg_type_and_index(&loop_op, position).map(|reg| reg.locale);
        // ⭐ Entry 0 is the induction variable's, which this island has no field for.
        assert_eq!(locale(0), None);
        assert_eq!(locale(1), Some(ops::RegType::Lar));
        assert_eq!(locale(2), Some(ops::RegType::Lbr));
        assert_eq!(locale(3), Some(ops::RegType::Lar));
        assert_eq!(locale(4), Some(ops::RegType::Lbr));
        assert_eq!(locale(5), None);
    }

    /// e136_fillTypedRegCollection — a promoted constant is the same in all units, a query map is so
    /// only when its per-unit constants agree, and a promoted value defined by anything else is
    /// DROPPED.
    #[test]
    fn e136_fill_typed_reg_collection() {
        let query_map = |second: i64| {
            vec![
                scalar_constant(Val(10), 5),
                scalar_constant(Val(11), second),
                Op::Uniform(uniform::Op::DefImmutableMapping {
                    result: Val(12),
                    pairs: vec![(Val(1), Val(10)), (Val(2), Val(11))],
                }),
                Op::Uniform(uniform::Op::QueryMap {
                    result: Val(13),
                    map: Val(12),
                    key: Val(3),
                }),
                scalar_copy(Val(14), ops::RegType::Lrf, None),
            ]
        };
        let fill = |init: Val, promoted: bool, second: i64| {
            let region = query_map(second);
            let regions: [&[Op]; 1] = [&region];
            let mut collection = TypedRegCollection::new(ops::RegType::Lrf);
            fill_typed_reg_collection(
                &mut collection,
                init,
                Val(20),
                Some(ops::RegIndex::at::<3>()),
                promoted,
                Definitions::from_innermost(&regions),
            );
            collection.regs
        };

        assert_eq!(
            fill(Val(10), true, 5),
            vec![TrackedReg::promoted(
                Val(20),
                Some(ops::RegIndex::at::<3>()),
                true
            )]
        );
        // Every unit's target constant is 5, so the register holds one value everywhere.
        assert_eq!(
            fill(Val(13), true, 5),
            vec![TrackedReg::promoted(
                Val(20),
                Some(ops::RegIndex::at::<3>()),
                true
            )]
        );
        assert_eq!(
            fill(Val(13), true, 6),
            vec![TrackedReg::promoted(
                Val(20),
                Some(ops::RegIndex::at::<3>()),
                false
            )]
        );
        assert_eq!(
            fill(Val(14), false, 5),
            vec![TrackedReg::new(Val(20), Some(ops::RegIndex::at::<3>()))]
        );
        // ⭐ THE SILENT DROP: promoted, but its initialiser is none of the three.
        assert_eq!(fill(Val(14), true, 5), Vec::new());
    }

    /// e348_updateRegIndex — the three answers: an unassigned register keeps its `-1`, a register
    /// numbered like an already-decided one inherits that decision, and anything else takes the
    /// suggestion and consumes it.
    #[test]
    fn e348_update_reg_index() {
        let mut collection = TypedRegCollection {
            reg_type: ops::RegType::Lrf,
            regs: vec![
                TrackedReg::new(Val(1), None),
                TrackedReg::new(Val(2), Some(ops::RegIndex::at::<7>())),
                TrackedReg::new(Val(3), Some(ops::RegIndex::at::<7>())),
            ],
        };

        assert!(!collection.update_reg_index(0, ops::RegIndex::at::<0>()));
        assert_eq!(collection.regs[0].new_index, None);

        assert!(collection.update_reg_index(1, ops::RegIndex::at::<0>()));
        assert_eq!(collection.regs[1].new_index, Some(ops::RegIndex::at::<0>()));

        // ⭐ SAME OLD INDEX AS THE ONE JUST DECIDED, so it follows it and `idx` is not consumed.
        assert!(!collection.update_reg_index(2, ops::RegIndex::at::<1>()));
        assert_eq!(collection.regs[2].new_index, Some(ops::RegIndex::at::<0>()));
    }

    /// e349_getRegWithType — an existing file is found where it sits, a new one is appended.
    #[test]
    fn e349_get_reg_with_type() {
        let mut reg_table = vec![
            TypedRegCollection::new(ops::RegType::Lrf),
            TypedRegCollection::new(ops::RegType::Lar),
        ];
        assert_eq!(reg_with_type(&mut reg_table, ops::RegType::Lar), 1);
        assert_eq!(reg_table.len(), 2);
        assert_eq!(reg_with_type(&mut reg_table, ops::RegType::Ebr), 2);
        assert_eq!(reg_table[2], TypedRegCollection::new(ops::RegType::Ebr));
    }
}
